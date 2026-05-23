import os
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from core.state import ResearchState

def construct_memory(state: ResearchState) -> dict:
    """
    Stage 6: Scientific Memory Construction.
    Stores papers and mechanisms in Vector DB (Qdrant).
    """
    chunks = state.get("mechanisms", [])
    print(f"--- STAGE 6: CONSTRUCTING MEMORY ---")
    print(f"Storing {len(chunks)} chunks in local Qdrant Vector Memory...")
    
    if not chunks:
        return {}
        
    os.makedirs("qdrant_data", exist_ok=True)
    client = QdrantClient(path="qdrant_data")
    
    collection_name = "scientific_memory"
    
    # Use lightweight local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_size = 384  # size of all-MiniLM-L6-v2
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
    # Embed and insert chunks
    print(f"Embedding {len(chunks)} chunks... This might take a moment.")
    vectors = embeddings.embed_documents(chunks)
    
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print("Successfully stored chunks in Qdrant.")
    return {}
