import streamlit as st
import os
import tempfile
import sys
import uuid

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import parse_resume
from src.utils import chunk_markdown
from src.vector_store import add_resume_to_db, query_resume
from src.llm_engine import get_llm_response

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("🚀 AI Resume Analyzer & Job Matcher")
st.markdown("### Build for free using Open Source Models (Qwen/Llama)")

# 2. Session Management (Multi-User Support)
# We generate a unique ID for every user so their data stays private
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4()}"

session_id = st.session_state.session_id

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type=["pdf"])
    
    st.info(f"Session ID: {session_id}")
    st.markdown("*Data is isolated to this session.*")

# --- Main Logic ---
if uploaded_file is not None:
    # 3. Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 4. Process Resume (Only if not already processed for this session)
    # We check if the file name matches to avoid re-parsing the same file
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Parsing Resume (Docling)..."):
            data = parse_resume(tmp_path)
            
            if data:
                st.session_state["resume_text"] = data["content"]
                st.session_state["current_file"] = uploaded_file.name
                
                # Chunk & Store in User-Specific Collection
                chunks = chunk_markdown(data["content"])
                add_resume_to_db(chunks, data["filename"], collection_name=session_id)
                st.success(f"Resume processed! ({len(chunks)} chunks stored)")
            else:
                st.error("Failed to parse resume.")

    # 5. Show Resume Preview
    with st.expander("📄 View Parsed Resume Content"):
        st.markdown(st.session_state.get("resume_text", ""))

    # --- Job Matching Section ---
    st.divider()
    st.header("2. Paste Job Description")
    job_description = st.text_area("Paste the JD here...", height=200)

    if st.button("Analyze Match"):
        if not job_description:
            st.warning("Please paste a Job Description first.")
        else:
            with st.spinner("🔍 Retrieving relevant experience..."):
                # RAG Step 1: Get Context from User's specific collection
                results = query_resume(job_description, collection_name=session_id, n_results=5)
                
                # Check if we actually found data
                if not results['documents'] or not results['documents'][0]:
                    st.error("No resume data found! Please upload a resume first.")
                    context_text = ""
                else:
                    context_text = "\n\n".join(results['documents'][0])
            
            if context_text:
                with st.spinner("🤖 AI Critiquing (Qwen-72B)..."):
                    # RAG Step 2: Ask LLM
                    response = get_llm_response(context_text, job_description)
                    
                    # Display Result
                    st.subheader("Analysis Result")
                    st.markdown(response)
                    
                    # Debug: Show what context was sent to LLM
                    with st.expander("See what the AI read (RAG Context)"):
                        st.info(context_text)

else:
    st.info("Please upload a PDF resume to start.")

# Cleanup temp file
if 'tmp_path' in locals() and os.path.exists(tmp_path):
    os.remove(tmp_path)