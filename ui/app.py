import streamlit as st
import os
import tempfile
import sys
import uuid
import re  # Added for robust score extraction

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import parse_resume
from src.utils import chunk_markdown
from src.vector_store import add_resume_to_db, query_resume
from src.llm_engine import get_llm_response

# 1. Page Configuration
st.set_page_config(page_title="AlignIQ AI", page_icon="🎯", layout="wide")

# Catchy Title & Design
st.title("🎯 AlignIQ AI")
st.subheader("Precision Career Alignment Engine")
# ADDED: Powered by note in header
st.caption("🚀 Intelligence powered by **Qwen 2.5 (72B Instruct)**")
st.markdown("---")

# 2. Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4()}"

# --- BUG FIX: Initialize a specific collection ID variable ---
if "collection_id" not in st.session_state:
    st.session_state.collection_id = st.session_state.session_id

session_id = st.session_state.session_id

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type=["pdf"])
    
    st.divider()
    st.info(f"**Session ID:** \n`{session_id}`")
    st.caption("Your data is isolated and will be cleared when the session ends.")

# --- Main Logic ---
if uploaded_file is not None:
    # 3. Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 4. Process Resume
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("🧠 Parsing Resume Structure..."):
            
            # --- BUG FIX START: Generate a FRESH ID for this specific file ---
            # This ensures we don't mix data from the previous upload
            unique_collection_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
            st.session_state.collection_id = unique_collection_id
            # ---------------------------------------------------------------
            
            data = parse_resume(tmp_path)
            
            if data:
                st.session_state["resume_text"] = data["content"]
                st.session_state["current_file"] = uploaded_file.name
                
                chunks = chunk_markdown(data["content"])
                
                # Use the UNIQUE collection ID, not just the generic session ID
                add_resume_to_db(chunks, data["filename"], collection_name=st.session_state.collection_id)
                
                st.success(f"✅ {uploaded_file.name} indexed successfully!")
            else:
                st.error("Failed to parse resume.")

    # 5. Show Resume Preview
    with st.expander("📄 View Parsed Resume Content"):
        st.markdown(st.session_state.get("resume_text", ""))

    # --- Job Matching Section ---
    st.header("📝 Analyze Career Alignment")
    job_description = st.text_area("Paste the Job Description (JD) here:", height=250, placeholder="Example: We are looking for a Senior AI Engineer...")

    if st.button("🚀 Run Match Analysis"):
        if not job_description:
            st.warning("Please paste a Job Description first.")
        else:
            with st.spinner("🔍 Retrieving relevant context from your resume..."):
                # --- BUG FIX: Query the specific collection ID ---
                results = query_resume(job_description, collection_name=st.session_state.collection_id, n_results=20)
                
                if not results['documents'] or not results['documents'][0]:
                    st.error("No resume data found! Please upload a resume first.")
                    context_text = ""
                else:
                    context_text = "\n\n".join(results['documents'][0])
            
            if context_text:
                with st.spinner("🤖 AI Recruiter is evaluating..."):
                    response = get_llm_response(context_text, job_description)
                    
                    # --- NEW DESIGN ELEMENTS START HERE ---
                    st.markdown("---")
                    
                    # 1. Extract Score for the Gauge
                    score_val = 0
                    try:
                        # Looks for "Match Score: 85" or "85/100"
                        match = re.search(r"Match Score:\s*(\d+)", response)
                        if match:
                            score_val = int(match.group(1))
                        else:
                            # Fallback: find any number followed by /100
                            match_alt = re.search(r"(\d+)/100", response)
                            if match_alt:
                                score_val = int(match_alt.group(1))
                    except:
                        score_val = 0

                    # 2. Display Metrics & Progress Bar
                    col_score, col_filler = st.columns([1, 2])
                    with col_score:
                        st.metric(label="Match Confidence", value=f"{score_val}%")
                    
                    st.progress(score_val / 100)
                    
                    # 3. Display the Full Report
                    st.subheader("📊 Comprehensive Analysis")
                    st.markdown(response)
                    
                    # 4. Context Expander
                    with st.expander("🛠 View RAG Source Context"):
                        st.info("The AI based its decision on these specific parts of your resume:")
                        st.write(context_text)
                    # --- NEW DESIGN ELEMENTS END HERE ---

else:
    st.info("👋 Welcome! Please upload your resume in the sidebar to begin.")

# Cleanup temp file (Windows fix)
try:
    if 'tmp_path' in locals() and os.path.exists(tmp_path):
        os.remove(tmp_path)
except PermissionError:
    pass