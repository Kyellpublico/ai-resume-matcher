import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

# We use a persistent client so data is saved to disk
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "resume_collection"

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    client = get_chroma_client()
    
    # Use standard open-source embedding model (free, runs locally)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    return collection

def add_resume_to_db(chunks, filename):
    """
    Takes chunks (from utils.py) and adds them to ChromaDB.
    """
    collection = get_collection()
    
    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        # chunk is a Document object from langchain splitter
        documents.append(chunk.page_content)
        
        # Merge existing metadata with our own
        meta = chunk.metadata.copy()
        meta["source"] = filename
        metadatas.append(meta)
        
        # Unique ID: filename + index
        ids.append(f"{filename}_{idx}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} chunks to ChromaDB.")

def query_resume(query_text, n_results=3):
    """
    Searches the DB for the most relevant resume sections.
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

def reset_db():
    """Clears the database (useful for testing)"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)