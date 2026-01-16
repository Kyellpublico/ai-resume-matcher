import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

CHROMA_PATH = "chroma_db"

def get_chroma_client():
    # Only initializing the client once is better practice
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(collection_name):
    """
    Creates or retrieves a collection specifically for a unique user session.
    """
    client = get_chroma_client()
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    return collection

def add_resume_to_db(chunks, filename, collection_name):
    collection = get_collection(collection_name)
    
    documents = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        meta = chunk.metadata.copy()
        meta["source"] = filename
        metadatas.append(meta)
        ids.append(f"{filename}_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

def query_resume(query_text, collection_name, n_results=3):
    collection = get_collection(collection_name)
    # Check if collection is empty to prevent crashes
    if collection.count() == 0:
        return {"documents": [[]]}
        
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results