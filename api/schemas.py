from pydantic import BaseModel
from typing import List, Optional

# --- Request Schemas ---
# What the user sends to us
class AnalyzeRequest(BaseModel):
    job_description: str

# --- Response Schemas ---
# What we send back to the user (Good for documentation)
class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str

class AnalyzeResponse(BaseModel):
    match_analysis: str
    context_used: str