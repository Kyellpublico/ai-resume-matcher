from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import sys

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import parse_resume
from src.utils import chunk_markdown
from src.vector_store import add_resume_to_db, query_resume
from src.llm_engine import get_llm_response

app = FastAPI(title="AI Resume Matcher API")

# Define the request body for analysis
class AnalyzeRequest(BaseModel):
    job_description: str

@app.get("/")
def home():
    return {"message": "AI Resume Matcher API is running"}

@app.post("/ingest")
async def ingest_resume(file: UploadFile = File(...)):
    """
    Uploads a PDF, parses it, chunks it, and saves to ChromaDB.
    """
    # 1. Save temp file
    temp_filename = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Parse
    data = parse_resume(temp_filename)
    if not data:
        raise HTTPException(status_code=400, detail="Failed to parse PDF")
        
    # 3. Chunk & Store
    chunks = chunk_markdown(data['content'])
    add_resume_to_db(chunks, data['filename'])
    
    # Clean up
    os.remove(temp_filename)
    
    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "status": "Success"
    }

@app.post("/analyze")
async def analyze_match(request: AnalyzeRequest):
    """
    Accepts a Job Description, retrieves resume context, and gets LLM feedback.
    """
    # 1. Retrieve Context (RAG)
    results = query_resume(request.job_description, n_results=5)
    
    if not results['documents'] or not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No resume data found. Please upload a resume first.")

    context_text = "\n\n".join(results['documents'][0])
    
    # 2. Get LLM Critique
    llm_response = get_llm_response(context_text, request.job_description)
    
    return {
        "match_analysis": llm_response,
        "context_used": context_text[:200] + "..." # Preview
    }