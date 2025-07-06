import streamlit as st
import tempfile
import os
from pathlib import Path
from smartqa.chunker import TextChunker
from smartqa.embedder import Embedder
from smartqa.retriever import Retriever
from smartqa.llm import LLMResponseGenerator
from smartqa.logger import QALogger


def setup_page():
    st.set_page_config(
        page_title="Smart Document QA",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Smart Document QA")
    st.markdown("Ask questions about your documents using AI-powered semantic search")


def upload_document():
    st.header("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt'],
        help="Upload a plain text document to analyze"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.success(f"✅ Document uploaded: {uploaded_file.name}")
        return tmp_file_path, uploaded_file.name
    
    return None, None


def process_document(file_path, file_name):
    with st.spinner("🔧 Processing document..."):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        with st.spinner("📄 Creating chunks..."):
            chunker = TextChunker()
            chunks = chunker.create_chunks(text)
        
        with st.spinner("🧠 Creating embeddings..."):
            embedder = Embedder()
            retriever = Retriever(embedder)
            retriever.add_chunks(chunks)
        
        st.success(f"✅ Document processed! Created {len(chunks)} chunks")
        
        return retriever, text, chunks


def display_document_info(text, chunks):
    st.subheader("📊 Document Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Characters", len(text))
    
    with col2:
        st.metric("Chunks", len(chunks))
    
    with col3:
        avg_chunk_size = len(text) // len(chunks) if chunks else 0
        st.metric("Avg Chunk Size", f"{avg_chunk_size} chars")
    
    with st.expander("🔍 View Document Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.text(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
            st.divider()


def ask_question(retriever, file_name):
    st.header("❓ Ask Questions")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of this document?"
    )
    
    if st.button("🤖 Ask", type="primary"):
        if question:
            with st.spinner("🔍 Searching for relevant information..."):
                search_results = retriever.search(question, k=3)
                
                if not search_results:
                    st.error("❌ No relevant information found")
                    return
                
                with st.spinner("🤖 Generating answer..."):
                    llm = LLMResponseGenerator()
                    response = llm.generate_response(question, search_results, file_name)
                
                st.subheader("📝 Answer")
                st.write(response["answer"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", response["tokens_used"])
                with col2:
                    st.metric("Response Time", f"{response.get('latency_ms', 0):.0f}ms")
                with col3:
                    st.metric("Citations", len(response["citations"]))
                
                if response["citations"]:
                    st.subheader("📚 Sources")
                    for i, citation in enumerate(response["citations"], 1):
                        with st.expander(f"Source {i} (Score: {citation['relevance_score']:.3f})"):
                            st.markdown(f"**Chunk ID:** {citation['chunk_id']}")
                            st.markdown(f"**Relevance Score:** {citation['relevance_score']:.3f}")
                            st.markdown("**Text:**")
                            st.text(citation['text'])
        else:
            st.warning("⚠️ Please enter a question")


def show_statistics():
    st.header("📊 System Statistics")
    
    logger = QALogger()
    stats = logger.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", stats["total_interactions"])
    
    with col2:
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
    
    with col3:
        st.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")
    
    with col4:
        st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
    
    if stats["total_interactions"] > 0:
        st.subheader("🕒 Recent Interactions")
        
        try:
            with open(logger.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_entries = lines[-5:][::-1]
            
            for line in recent_entries:
                if line.strip():
                    import json
                    entry = json.loads(line)
                    
                    with st.expander(f"Q: {entry['query'][:50]}..."):
                        st.markdown(f"**Question:** {entry['query']}")
                        st.markdown(f"**Answer:** {entry['answer']}")
                        st.markdown(f"**Tokens:** {entry['tokens_used']} | **Time:** {entry['latency_ms']:.0f}ms")
                        st.markdown(f"**File:** {entry['input_file']} | **Model:** {entry['model_name']}")
                        st.markdown(f"**Date:** {entry['timestamp']}")
        except Exception as e:
            st.error(f"Error reading log file: {e}")


def main():
    setup_page()
    
    file_path, file_name = upload_document()
    
    if file_path:
        retriever, text, chunks = process_document(file_path, file_name)
        
        display_document_info(text, chunks)
        
        ask_question(retriever, file_name)
        
        show_statistics()
        
        try:
            os.unlink(file_path)
        except:
            pass
    else:
        st.info("👆 Please upload a document to get started")
        
        show_statistics()


if __name__ == "__main__":
    main() 