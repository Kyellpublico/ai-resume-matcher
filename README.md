# 🎯AlignIQ AI

A professional **RAG (Retrieval-Augmented Generation)** application that helps job seekers tailor their resumes to specific job descriptions. 

It uses **Docling** for advanced PDF parsing, **ChromaDB** for semantic search, and **Qwen-2.5-72B** (via Hugging Face) to act as an expert technical recruiter.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-purple)

## 🏗 System Architecture

The project follows a modular **Microservice RAG pipeline**:

1.  **Ingestion:** PDF resumes are parsed using **IBM Docling** to preserve document structure.
2.  **Chunking:** Text is split into semantic chunks using `MarkdownHeaderTextSplitter`.
3.  **Embedding:** vectorized using `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Multi-Tenant Storage:** Vectors are stored in **ChromaDB** with **Session-Based Isolation** (User A cannot see User B's data).
5.  **Retrieval:** The system performs a semantic search to find the top 5 resume sections relevant to the Job Description.
6.  **Analysis:** The retrieved context + Job Description are sent to **Qwen-2.5-72B**, which provides a 0-100 match score and gap analysis.

## ✨ Key Features
* **🔒 Multi-Tenancy:** built-in session management ensures data privacy between different users.
* **Dual Interface:**
    * **Web UI (Streamlit):** For interactive testing and visual feedback.
    * **REST API (FastAPI):** For programmatic access and integration.
* **Advanced Parsing:** Handles complex PDF layouts (tables/columns) using Docling.
* **Zero-Cost Stack:** Runs entirely on free tiers (Hugging Face Serverless API) and open-source local libraries.

## 🛠 Tech Stack
* **LLM:** Qwen-2.5-72B-Instruct (via Hugging Face Inference API)
* **Vector DB:** ChromaDB (Persistent Local Storage)
* **Backend:** FastAPI + Pydantic
* **Frontend:** Streamlit
* **Tools:** Docling, LangChain, Uvicorn, Python-Dotenv

## 📂 Project Structure

```text
ai-resume-matcher/
├── data/                   # Temporary storage for uploaded files
├── notebooks/              # Jupyter notebooks for R&D (Experiments)
├── src/                    # Core Business Logic
│   ├── ingestion.py        # PDF parsing logic (Docling)
│   ├── vector_store.py     # ChromaDB interface
│   ├── llm_engine.py       # Hugging Face API connection
│   └── utils.py            # Text chunking helpers
├── api/                    # Production API
│   ├── main.py             # FastAPI Endpoints
│   └── schemas.py          # Pydantic Data Models
├── ui/                     # User Interface
│   └── app.py              # Streamlit Dashboard
├── .env                    # Environment variables (Secrets)
└── README.md               # Documentation
```
## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Kyellpublico/ai-resume-matcher.git
cd ai-resume-matcher
```

### 2. Set up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.env` file in the root directory and add your Hugging Face Token:
```Ini
HF_TOKEN=hf_YourHuggingFaceTokenHere
```

### 4. Run the Application
**Option A: Interactive Dashboard (Streamlit)**
Best for testing the workflow visually.
```bash
streamlit run ui/app.py
```
**Option B: REST API (FastAPI)**
Best for backend development or integration.
```bash
uvicorn api.main:app --reload
```
- **API Docs:** Open ```http://127.0.0.1:8000/docs``` to test endpoints via Swagger UI.

## 🧪 Usage Workflow
1. **Upload Resume:** Upload your PDF resume. The system parses and indexes it into ChromaDB.
2. **Paste Job Description:** Find a job posting you are interested in.
3. **Get AnalyGet Analysis:** The AI will generate:sis: The AI will generate:
    - **Match Score (0-100)**
    - **Gap Analysis** (Missing Skills)
    - **Tailoring Advice** (How to fix your resume)