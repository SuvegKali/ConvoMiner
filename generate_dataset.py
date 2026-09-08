import json
import random
import os
from datetime import datetime, timedelta
from openai import OpenAI

# Initialize OpenAI client (uses OPENAI_API_KEY environment variable)
client = OpenAI(
    api_key="sk-air-v1-d6d0a428d331befa59f5117be004d29fca6b38aca3cc008bc7f13de879aa367b",                # your AIRouter API key
    base_url="https://api.airouter.in/v1",  # AIRouter's OpenAI-compatible endpoint
)

PARTICIPANTS = [
    {"id": "p1", "name": "Priya"},
    {"id": "p2", "name": "Rohan"},
    {"id": "p3", "name": "Ananya"},
    {"id": "p4", "name": "Kabir"},
    {"id": "p5", "name": "Aarav"},
    {"id": "p6", "name": "Sneha"},
    {"id": "p7", "name": "Vikram"},
    {"id": "p8", "name": "Diya"}
]

FILLER_POOL = [
    "haan bhai", "sahi hai", "lol", "kya?", "chalo done", "aaj nahi ho payega",
    "good morning guys", "bc kya chal raha hai", "ok", "aacha aisa kya?",
    "send link", "ha ha ha", "bhai match dekha?", "kaha ho sab?", "yurrr",
    "brb 10 mins", "kal baat karte hain", "oye answer de", "ha perfect",
    "bro 5 mins me call karta hu", "haha dead 😂", "aaj shaam ko milte hain?",
    "zomato se kya mangaye?", "pata nahi yaar", "batao batao", "haan haan"
]

ANCHOR_PROMPTS = [
    {
        "topic": "Planning a Manali trip",
        "decision": "Deciding on going to Manali instead of Goa, setting dates, and booking a homestay.",
        "zero_keyword_decision": "chalo Manali fix hai phir"
    },
    {
        "topic": "Flat hunting and rent split",
        "decision": "Priya agreeing to pay max 5k for room deposit and Vikram handling security.",
        "zero_keyword_decision": "max 5k per person bro usse zyada nahi"
    },
    {
        "topic": "Surprise birthday gift for Kabir",
        "decision": "Buying a PS5 controller instead of a smartwatch.",
        "zero_keyword_decision": "gamepad mangwa lete hain, watch rehne do"
    }
]

def generate_anchor_thread(prompt_info):
    """Uses LLM to generate a realistic Hinglish sub-conversation thread."""
    system_prompt = (
        "You are an expert at generating hyper-realistic Indian WhatsApp group chat conversations. "
        "Write in natural Hinglish (mix of English and Hindi transliteration) with informal slang, typos, "
        "short messages, and realistic dynamics between 8 friends."
    )
    user_prompt = f"""
    Generate a realistic chat thread about: '{prompt_info['topic']}'.
    Participants: {', '.join([p['name'] for p in PARTICIPANTS])}.
    The thread must include a key decision point where someone types a message like: '{prompt_info['zero_keyword_decision']}'.
    
    Output JSON array of objects format:
    [
      {{"sender": "Priya", "content": "message here"}},
      {{"sender": "Rohan", "content": "message here"}}
    ]
    Generate between 25 to 35 messages. Output ONLY raw JSON without markdown backticks.
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(content)

def generate_test_queries():
    """Generates 40 query test cases matching the problem statement requirements."""
    prompt = """
    Generate 40 evaluation queries for a group chat search system in JSON.
    Query types: 'semantic' (15 queries), 'attributed' (15 queries), 'temporal' (10 queries).
    At least 8 semantic queries MUST NOT share any exact keywords with the answer.
    
    Output format:
    [
      {
        "query_id": "q1",
        "type": "semantic",
        "query": "when did we decide on the trip destination?",
        "notes": "Target message uses zero keywords (e.g. 'chalo Manali fix hai phir')"
      }
    ]
    Output ONLY valid JSON.
    """
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(content)

def assemble_dataset():
    print("Generating anchor threads via API...")
    anchor_threads = [generate_anchor_thread(p) for p in ANCHOR_PROMPTS]
    
    print("Generating 40 test queries...")
    test_queries = generate_test_queries()

    start_date = datetime(2026, 3, 1, 9, 0, 0)
    total_target_messages = 4000
    
    messages = []
    current_time = start_date
    msg_id = 1

    # Inject anchor threads at message indices 800, 2000, 3200
    anchor_positions = {800: anchor_threads[0], 2000: anchor_threads[1], 3200: anchor_threads[2]}
    
    print("Building full 4,000-message timeline...")
    while msg_id <= total_target_messages:
        # Check if we hit an anchor point
        if msg_id in anchor_positions:
            thread = anchor_positions[msg_id]
            for item in thread:
                participant = next((p for p in PARTICIPANTS if p['name'] == item['sender']), PARTICIPANTS[0])
                messages.append({
                    "id": msg_id,
                    "sender_id": participant["id"],
                    "sender_name": participant["name"],
                    "timestamp": current_time.isoformat() + "Z",
                    "content": item["content"]
                })
                current_time += timedelta(seconds=random.randint(10, 180))
                msg_id += 1
        else:
            participant = random.choice(PARTICIPANTS)
            content = random.choice(FILLER_POOL)
            messages.append({
                "id": msg_id,
                "sender_id": participant["id"],
                "sender_name": participant["name"],
                "timestamp": current_time.isoformat() + "Z",
                "content": content
            })
            # Random time jump (15 secs to 3 hours)
            current_time += timedelta(seconds=random.randint(15, 10800))
            msg_id += 1

    # Link test queries to exact message IDs
    test_queries[0]["target_message_id"] = 812  # Near anchor 1 decision
    test_queries[1]["target_message_id"] = 2015 # Near anchor 2 decision

    dataset = {
        "dataset_info": {
            "id": "set_01",
            "created_at": datetime.now().isoformat(),
            "total_messages": len(messages),
            "description": "Synthetic 6-month Hinglish chat with 8 users and 3 core decisions."
        },
        "messages": messages,
        "test_queries": test_queries
    }

    os.makedirs("datasets", exist_ok=True)
    with open("datasets/dataset_1.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset generated successfully! Total messages: {len(messages)}. Saved to datasets/dataset_1.json")

if __name__ == "__main__":
    assemble_dataset()