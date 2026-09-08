import sqlite3
import json
import chromadb
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

# Initialize clients
client = OpenAI(
    api_key="sk-air-v1-d6d0a428d331befa59f5117be004d29fca6b38aca3cc008bc7f13de879aa367b",                # your AIRouter API key
    base_url="https://api.airouter.in/v1",  # AIRouter's OpenAI-compatible endpoint
)
chroma_client = chromadb.PersistentClient(path="./db/chroma_data")
collection = chroma_client.get_collection("chat_search")

DB_PATH = "./db/chat.db"

class ParsedQuery(BaseModel):
    semantic_query: str
    sender_filter: Optional[str] = None
    start_date: Optional[str] = None  # ISO format string YYYY-MM-DD
    end_date: Optional[str] = None    # ISO format string YYYY-MM-DD

def parse_query_intent(user_query: str) -> ParsedQuery:
    """Uses LLM structured output to extract metadata filters and query keywords."""
    system_prompt = """
    You are a query parsing router for a group chat search system.
    Extract the following from the user's natural language search query:
    1. semantic_query: Cleaned topic/keywords stripped of sender names or relative temporal words. Translate Hinglish concepts if necessary for better semantic search.
    2. sender_filter: Exact sender name mentioned (e.g. 'Priya', 'Rohan', 'Kabir') or null.
    3. start_date / end_date: Strictly format as YYYY-MM-DD or null if no temporal constraint is mentioned. Assume current date is 2026-09-01.

    Output pure JSON matching the requested schema.
    """
    
    response = client.beta.chat.completions.parse(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        response_format=ParsedQuery
    )
    return response.choices[0].message.parsed

def fetch_conversation_window(target_msg_id: int, window: int = 5) -> list:
    """Fetches target message along with surrounding conversation context from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    start_id = max(1, target_msg_id - window)
    end_id = target_msg_id + window
    
    cursor.execute(
        "SELECT id, sender_name, timestamp, content FROM messages WHERE id BETWEEN ? AND ? ORDER BY id ASC",
        (start_id, end_id)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def execute_search(user_query: str, top_k: int = 3):
    # Step 1: Parse intent
    parsed = parse_query_intent(user_query)
    print(f"Parsed Router Intent: {parsed}")

    # Step 2: Build ChromaDB metadata filter
    where_conditions = []
    if parsed.sender_filter:
        where_conditions.append({"sender_name": {"$eq": parsed.sender_filter}})
    if parsed.start_date:
        where_conditions.append({"timestamp": {"$gte": parsed.start_date}})
    if parsed.end_date:
        where_conditions.append({"timestamp": {"$lte": parsed.end_date}})

    chroma_where = None
    if len(where_conditions) == 1:
        chroma_where = where_conditions[0]
    elif len(where_conditions) > 1:
        chroma_where = {"$and": where_conditions}

    # Step 3: Embed the cleaned semantic string
    query_embedding = client.embeddings.create(
        input=[parsed.semantic_query],
        model="openai/text-embedding-3-small"
    ).data[0].embedding

    # Step 4: Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=chroma_where
    )

    if not results["ids"][0]:
        return {"router_intent": parsed.model_dump(), "results": []}

    # Step 5: Fetch context windows from SQLite
    formatted_results = []
    for i in range(len(results["ids"][0])):
        match_id = int(results["ids"][0][i])
        distance = results["distances"][0][i]
        
        context_window = fetch_conversation_window(match_id, window=5)
        
        formatted_results.append({
            "matched_message_id": match_id,
            "similarity_score": round(1 - distance, 4),
            "context_window": context_window
        })

    return {
        "router_intent": parsed.model_dump(),
        "results": formatted_results
    }

if __name__ == "__main__":
    # Test execution
    test_res = execute_search("what did Priya say about the budget")
    print(json.dumps(test_res, indent=2))