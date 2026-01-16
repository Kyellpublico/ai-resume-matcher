from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Logic
from src.ingestion import parse_resume
from src.utils import chunk_markdown
from src.vector_store import add_resume_to_db, query_resume
from src.llm_engine import get_llm_response

# Import Schemas (The new part!)
from .schemas import AnalyzeRequest, IngestResponse, AnalyzeResponse

app = FastAPI(title="AI Resume Matcher API")

@app.get("/")
def home():
    return {"message": "AI Resume Matcher API is running"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_resume(file: UploadFile = File(...)):
    """
    Uploads a PDF, parses it, chunks it, and saves to ChromaDB.
    """
    temp_filename = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        data = parse_resume(temp_filename)
        if not data:
            raise HTTPException(status_code=400, detail="Failed to parse PDF")
            
        chunks = chunk_markdown(data['content'])
        add_resume_to_db(chunks, data['filename'])
        
        return IngestResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            status="Success"
        )
    finally:
        # Always clean up, even if error occurs
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_match(request: AnalyzeRequest):
    """
    Accepts a Job Description, retrieves resume context, and gets LLM feedback.
    """
    results = query_resume(request.job_description, n_results=5)
    
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No resume data found. Please upload a resume first.")

    context_text = "\n\n".join(results['documents'][0])
    
    llm_response = get_llm_response(context_text, request.job_description)
    
    return AnalyzeResponse(
        match_analysis=llm_response,
        context_used=context_text[:500] + "..." # Limit preview size
    )