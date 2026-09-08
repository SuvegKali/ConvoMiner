import json
import sqlite3
import os
import chromadb
from openai import OpenAI

# Initialize clients
client = OpenAI(
    api_key="sk-air-v1-d6d0a428d331befa59f5117be004d29fca6b38aca3cc008bc7f13de879aa367b",                # your AIRouter API key
    base_url="https://api.airouter.in/v1",  # AIRouter's OpenAI-compatible endpoint
)
chroma_client = chromadb.PersistentClient(path="./db/chroma_data")

DB_PATH = "./db/chat.db"

def init_sqlite():
    """Sets up SQLite database with clean table structure."""
    os.makedirs("./db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing table for clean reset during dataset swaps
    cursor.execute("DROP TABLE IF EXISTS messages")
    cursor.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            sender_id TEXT,
            sender_name TEXT,
            timestamp TEXT,
            content TEXT
        )
    """)
    conn.commit()
    return conn

def get_contextual_text(messages, index, window_size=2):
    """
    Builds context window string for target message.
    Example output: 'Priya: Where to go? | Rohan: Let's do north | Kabir: chalo Manali fix hai'
    """
    start_idx = max(0, index - window_size)
    context_msgs = messages[start_idx : index + 1]
    
    formatted_pieces = []
    for m in context_msgs:
        formatted_pieces.append(f"{m['sender_name']}: {m['content']}")
        
    return " | ".join(formatted_pieces)

def ingest_dataset(json_file_path):
    print(f"Loading {json_file_path}...")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data["messages"]
    
    # 1. Populate SQLite
    conn = init_sqlite()
    cursor = conn.cursor()
    
    db_records = [
        (m["id"], m["sender_id"], m["sender_name"], m["timestamp"], m["content"])
        for m in messages
    ]
    
    cursor.executemany(
        "INSERT INTO messages (id, sender_id, sender_name, timestamp, content) VALUES (?, ?, ?, ?, ?)",
        db_records
    )
    conn.commit()
    conn.close()
    print(f"Inserted {len(messages)} messages into SQLite.")

    # 2. Reset and populate ChromaDB Collection
    try:
        chroma_client.delete_collection("chat_search")
    except Exception:
        pass  # Collection didn't exist yet

    collection = chroma_client.create_collection(
        name="chat_search",
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Batch process embeddings with contextual text
    batch_size = 200
    print("Generating contextual embeddings and populating ChromaDB...")

    for i in range(0, len(messages), batch_size):
        batch = messages[i : i + batch_size]
        
        ids = [str(m["id"]) for m in batch]
        metadatas = [
            {
                "message_id": m["id"],
                "sender_name": m["sender_name"],
                "timestamp": m["timestamp"]
            }
            for m in batch
        ]
        
        # Build contextual text strings for embedding
        documents = [
            get_contextual_text(messages, i + idx, window_size=2)
            for idx, m in enumerate(batch)
        ]

        # Call OpenAI Embeddings API
        response = client.embeddings.create(
            input=documents,
            model="openai/text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed messages {i + 1} to {min(i + batch_size, len(messages))}...")

    print("Ingestion complete! Database ready for queries.")

if __name__ == "__main__":
    ingest_dataset("datasets/dataset_1.json")