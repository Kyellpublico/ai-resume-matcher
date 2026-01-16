from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
import os
import sys
import uuid

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import parse_resume
from src.utils import chunk_markdown
# Note: We updated these functions in Part 1 to accept collection_name
from src.vector_store import add_resume_to_db, query_resume 
from src.llm_engine import get_llm_response
from .schemas import AnalyzeRequest, IngestResponse, AnalyzeResponse

app = FastAPI(title="AI Resume Matcher API")

@app.get("/")
def home():
    return {"message": "AI Resume Matcher API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_resume(
    file: UploadFile = File(...), 
    session_id: str = Form(None) # <--- Accept session_id as form data (Optional)
):
    """
    Uploads a PDF. If no session_id is provided, generates a new one.
    """
    # 1. Generate Session ID if missing
    if not session_id:
        session_id = f"session_{uuid.uuid4()}"
    
    temp_filename = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        data = parse_resume(temp_filename)
        if not data:
            raise HTTPException(status_code=400, detail="Failed to parse PDF")
            
        chunks = chunk_markdown(data['content'])
        
        # 2. Store in User-Specific Collection
        add_resume_to_db(chunks, data['filename'], collection_name=session_id)
        
        return IngestResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            status="Success",
            session_id=session_id  # Return ID so client can use it next time
        )
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_match(request: AnalyzeRequest):
    """
    Analyzes match based on the specific session_id provided.
    """
    # 1. Query the Specific User's Collection
    results = query_resume(
        request.job_description, 
        collection_name=request.session_id,  # <--- Use the session_id from request
        n_results=5
    )
    
    # Check if we found anything (if list is empty or first element is empty)
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(
            status_code=404, 
            detail=f"No resume data found for session ID: {request.session_id}"
        )

    context_text = "\n\n".join(results['documents'][0])
    
    llm_response = get_llm_response(context_text, request.job_description)
    
    return AnalyzeResponse(
        match_analysis=llm_response,
        context_used=context_text[:500] + "..."
    )