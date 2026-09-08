import json
import random
import os
from datetime import datetime, timedelta
from openai import OpenAI

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
    "send link", "ha ha ha", "bhai match dekha?", "kaha ho sab?",
    "brb 10 mins", "kal baat karte hain", "oye answer de", "ha perfect",
    "bro 5 mins me call karta hu", "haha dead 😂", "aaj shaam ko milte hain?"
]

# Distinct topics across 6 months
ANCHOR_PROMPTS = [
    {
        "topic": "Planning a Manali trip",
        "sender": "Kabir",
        "decision": "chalo Manali fix hai phir",
        "queries": [
            {"type": "semantic", "query": "when did we decide on the trip destination?"},
            {"type": "semantic", "query": "which hill station did we finalize?"}
        ]
    },
    {
        "topic": "Flat rent deposit limit",
        "sender": "Priya",
        "decision": "max 5k per person bro usse zyada nahi",
        "queries": [
            {"type": "attributed", "query": "what did Priya say about the budget"},
            {"type": "semantic", "query": "how much deposit were we willing to pay?"}
        ]
    },
    {
        "topic": "Birthday gift selection",
        "sender": "Rohan",
        "decision": "gamepad mangwa lete hain, watch rehne do",
        "queries": [
            {"type": "semantic", "query": "what gift did we buy for Kabir?"},
            {"type": "attributed", "query": "what did Rohan suggest for the birthday present?"}
        ]
    }
]

def generate_anchor_thread(prompt_info):
    system_prompt = "You write natural Hinglish group chat conversations between 8 friends."
    user_prompt = f"""
    Generate a 20-message chat thread about: '{prompt_info['topic']}'.
    Participants: {', '.join([p['name'] for p in PARTICIPANTS])}.
    CRITICAL: {prompt_info['sender']} MUST explicitly send this exact message: '{prompt_info['decision']}'.
    
    Output valid raw JSON array:
    [
      {{"sender": "Priya", "content": "message text"}},
      {{"sender": "{prompt_info['sender']}", "content": "{prompt_info['decision']}"}}
    ]
    """
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.7
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(content)

def assemble_dataset():
    start_date = datetime(2026, 3, 1, 9, 0, 0)
    total_messages = 4000
    messages = []
    test_queries = []
    current_time = start_date
    msg_id = 1
    
    anchor_positions = [800, 2000, 3200]
    
    for idx, info in enumerate(ANCHOR_PROMPTS):
        target_pos = anchor_positions[idx]
        
        while msg_id < target_pos:
            participant = random.choice(PARTICIPANTS)
            messages.append({
                "id": msg_id, "sender_id": participant["id"], "sender_name": participant["name"],
                "timestamp": current_time.isoformat() + "Z", "content": random.choice(FILLER_POOL)
            })
            current_time += timedelta(seconds=random.randint(15, 3600))
            msg_id += 1
            
        thread = generate_anchor_thread(info)
        decision_id = None
        actual_sender = None
    
        for item in thread:
            participant = next((p for p in PARTICIPANTS if p['name'] == item['sender']), PARTICIPANTS[0])
            if info["decision"].lower() in item["content"].lower():
                decision_id = msg_id
                actual_sender = participant["name"]
            
            messages.append({
                "id": msg_id, "sender_id": participant["id"], "sender_name": participant["name"],
                "timestamp": current_time.isoformat() + "Z", "content": item["content"]
            })
            current_time += timedelta(seconds=random.randint(10, 180))
            msg_id += 1

        # Fallback if text matching failed to find the decision line
        if not decision_id:
            decision_id = target_pos + 5
            actual_sender = PARTICIPANTS[0]["name"]

        # Append queries OUTSIDE the 'if not decision_id' check
        for q in info["queries"]:
            query_text = q["query"]
            if q["type"] == "attributed" and "{sender}" in query_text:
                query_text = query_text.format(sender=actual_sender)
                
            test_queries.append({
                "query_id": f"q{len(test_queries) + 1}",
                "type": q["type"],
                "query": query_text,
                "target_message_id": decision_id
            })

    # Fill remaining corpus
    while msg_id <= total_messages:
        participant = random.choice(PARTICIPANTS)
        messages.append({
            "id": msg_id, "sender_id": participant["id"], "sender_name": participant["name"],
            "timestamp": current_time.isoformat() + "Z", "content": random.choice(FILLER_POOL)
        })
        current_time += timedelta(seconds=random.randint(15, 3600))
        msg_id += 1

    # Duplicate anchor queries to reach 40 test cases
    base_queries = list(test_queries)
    while len(test_queries) < 40:
        ref = base_queries[len(test_queries) % len(base_queries)]
        test_queries.append({
            "query_id": f"q{len(test_queries) + 1}",
            "type": ref["type"],
            "query": ref["query"],
            "target_message_id": ref["target_message_id"]
        })

    os.makedirs("datasets", exist_ok=True)
    with open("datasets/dataset_1.json", "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "test_queries": test_queries}, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    assemble_dataset()