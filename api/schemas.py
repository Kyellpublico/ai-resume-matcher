from pydantic import BaseModel
from typing import Optional

# --- Request Schemas ---
class AnalyzeRequest(BaseModel):
    job_description: str
    session_id: str  # <--- NEW: Required to know WHICH resume to analyze

# --- Response Schemas ---
class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str
    session_id: str # <--- NEW: We return this so the frontend can save it

class AnalyzeResponse(BaseModel):
    match_analysis: str
    context_used: str