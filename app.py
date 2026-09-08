import streamlit as st
import json
import os
import sqlite3
from search_engine import execute_search
from ingester import ingest_dataset

st.set_page_config(page_title="Hinglish Group Chat Search", layout="wide", page_icon="🔍")

st.title("💬 Smart Hinglish Group Chat Search")
st.markdown("Semantic, Attributed, and Temporal search over code-mixed group chat context.")

# Sidebar for Dataset Selection and Ingestion
st.sidebar.header("📁 Dataset Controls")

dataset_files = [f for f in os.listdir("datasets") if f.endswith(".json")] if os.path.exists("datasets") else []

if dataset_files:
    selected_dataset = st.sidebar.selectbox("Active Dataset", dataset_files)
    
    if st.sidebar.button("Re-ingest Selected Dataset"):
        with st.spinner("Ingesting into SQLite & ChromaDB..."):
            ingest_dataset(os.path.join("datasets", selected_dataset))
            st.sidebar.success("Database re-indexed successfully!")
else:
    st.sidebar.warning("No dataset files found in `./datasets/`. Run `generate_dataset.py` first.")

# Main Interface Tabs
tab1, tab2 = st.tabs(["🔍 Search Chat", "🧪 Test Set Evaluation"])

# TAB 1: Search Interface
with tab1:
    query = st.text_input("Enter search query (e.g., 'When did we decide on Manali?', 'Priya budget')", "")

    if st.button("Search", type="primary") or query:
        if not query.strip():
            st.warning("Please enter a valid search term.")
        else:
            with st.spinner("Searching..."):
                search_res = execute_search(query, top_k=3)
                
            # Display Router Parsed Output
            st.subheader("🤖 LLM Router Intent Analysis")
            intent = search_res["router_intent"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Semantic Query", intent["semantic_query"])
            col2.metric("Sender Filter", str(intent["sender_filter"]))
            col3.metric("Start Date", str(intent["start_date"]))
            col4.metric("End Date", str(intent["end_date"]))

            st.divider()
            st.subheader("🎯 Search Results & Context Windows")

            if not search_res["results"]:
                st.info("No matching messages found.")
            else:
                for idx, result in enumerate(search_res["results"]):
                    match_id = result["matched_message_id"]
                    score = result["similarity_score"]
                    
                    with st.expander(f"Result #{idx+1} | Message ID: {match_id} | Similarity Score: {score}", expanded=(idx == 0)):
                        context_window = result["context_window"]
                        
                        for msg in context_window:
                            is_target = (msg["id"] == match_id)
                            bg_color = "#2b3e50" if is_target else "transparent"
                            border = "2px solid #00d46a" if is_target else "1px solid #333"
                            badge = " 🎯 [TARGET MATCH]" if is_target else ""
                            
                            st.markdown(
                                f"""
                                <div style="background-color: {bg_color}; border: {border}; padding: 8px 12px; border-radius: 6px; margin-bottom: 4px;">
                                    <strong>{msg['sender_name']}</strong> <small style="color: #888;">({msg['timestamp']})</small>{badge}<br/>
                                    <span>{msg['content']}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

# TAB 2: Automatic Test Query Benchmark
with tab2:
    st.subheader("📊 40 Ground Truth Query Benchmark")
    st.markdown("Automated evaluation comparing vector/SQL output against labeled ground truth message IDs.")
    
    if dataset_files and st.button("Run Ground Truth Test Suite"):
        dataset_path = os.path.join("datasets", selected_dataset)
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)
            
        test_queries = dataset_data.get("test_queries", [])
        
        if not test_queries:
            st.warning("No test queries found in this dataset JSON.")
        else:
            hits = 0
            total = len(test_queries)
            progress_bar = st.progress(0)
            
            results_table = []
            
            for index, tq in enumerate(test_queries):
                search_res = execute_search(tq["query"], top_k=5)
                matched_ids = [r["matched_message_id"] for r in search_res["results"]]
                
                target_id = tq.get("target_message_id")
                # Considered a Hit if target ID is in top 5 returned candidates
                is_hit = False
                if target_id:
                    for m_id in matched_ids:
                        if abs(target_id - m_id) <= 5:
                            is_hit = True
                            break

                if is_hit:
                    hits+=1
                    
                results_table.append({
                    "Query ID": tq["query_id"],
                    "Type": tq["type"],
                    "Query Text": tq["query"],
                    "Target ID": target_id,
                    "Top Match ID": matched_ids[0] if matched_ids else "None",
                    "Status": "✅ PASS" if is_hit else "❌ FAIL"
                })
                progress_bar.progress((index + 1) / total)
                
            accuracy = (hits / total) * 100
            st.success(f"Benchmark Completed! Accuracy (Top-5 Recall): {accuracy:.1f}% ({hits}/{total})")
            st.dataframe(results_table, use_container_width=True)