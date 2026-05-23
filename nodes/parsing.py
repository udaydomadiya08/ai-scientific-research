import os
import requests
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.state import ResearchState

def parse_and_chunk(state: ResearchState) -> dict:
    """
    Stage 4 & 5: Autonomous Paper Reading & Semantic Chunking.
    Downloads PDFs, extracts text, and chunks them semantically.
    """
    papers = state.get("retrieved_papers", [])
    print(f"--- STAGE 4 & 5: PARSING & CHUNKING {len(papers)} PAPERS ---")
    
    os.makedirs("data", exist_ok=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    
    for idx, p in enumerate(papers):
        url = p.get("url")
        if not url:
            continue
            
        print(f"Downloading {p.get('title')}...")
        try:
            # Arxiv urls usually end in .pdf or we can just append it
            pdf_url = url + ".pdf" if not url.endswith(".pdf") else url
            response = requests.get(pdf_url, timeout=10)
            
            pdf_path = f"data/paper_{idx}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
                
            # Extract text
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
                
            print(f"Extracted {len(text)} characters. Chunking...")
            chunks = text_splitter.split_text(text)
            
            # Keep track of source
            for c in chunks:
                all_chunks.append(f"[Source: {p.get('title')}]\n{c}")
                
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            
    return {"mechanisms": all_chunks}
