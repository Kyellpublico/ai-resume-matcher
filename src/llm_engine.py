import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

def get_llm_response(resume_context, job_description):
    """
    Sends the resume context and JD to the LLM for analysis.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Error: HF_TOKEN not found in .env file."

    # Initialize the client (Free Serverless API)
    # We use Qwen 2.5 72B (huge model!) for best reasoning
    client = InferenceClient(
        model="Qwen/Qwen2.5-72B-Instruct", 
        token=token
    )

    # The Prompt Engineering
    prompt = f"""
    You are an expert Senior Technical Recruiter. 
    
    JOB DESCRIPTION:
    {job_description}
    
    RESUME CONTEXT (Retrieved parts of the candidate's resume):
    {resume_context}
    
    TASK:
    1. Analyze how well the resume matches the job description based *only* on the context provided.
    2. Provide a Match Score (0-100).
    3. List "Missing Keywords" or "Gap Analysis".
    4. Provide specific advice to improve the resume for this specific job.
    
    OUTPUT FORMAT:
    **Match Score:** [Score]/100
    
    **Analysis:**
    [Your analysis here]
    
    **Missing Critical Skills:**
    - [Skill 1]
    - [Skill 2]
    
    **Advice:**
    [Advice here]
    """

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=False
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling LLM: {e}"

# --- Test Block ---
if __name__ == "__main__":
    # Fake data to test the connection
    fake_context = "I have 3 years of experience in Python, FastAPI, and Docker."
    fake_jd = "Looking for a Senior Python Developer with Kubernetes experience."
    
    print("Asking the AI...")
    print(get_llm_response(fake_context, fake_jd))