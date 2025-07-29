from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List
import uuid

from app.services.pdf_processor import PDFProcessor
from app.db.vector_db import VectorDB
from app.services.summarizer import Summarizer
from app.models.schemas import PDFUpload, SummaryRequest, SummaryResponse, AskRequest
from app.utils.config import settings

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
vector_db = VectorDB()
summarizer = Summarizer()

DATA_DIR = "data"
Path(DATA_DIR).mkdir(exist_ok=True)

@app.post("/upload", response_model=PDFUpload)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    return {"filename": file.filename}

@app.post("/process/{filename}", response_model=List[str])
async def process_pdf(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    print("file path :", file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Process PDF
    chunks = pdf_processor.extract_text(file_path)
    
    # Generate unique IDs for each chunk
    doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    
    # Store in vector DB
    collection_name = filename.split(".")[0]
    vector_db.create_collection(collection_name, chunks, doc_ids)
    
    return doc_ids

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummaryRequest):
    # In a real application, you would retrieve the document from the vector DB
    # For simplicity, we'll just summarize the first chunk
    collection = vector_db.get_collection(request.document_id.split("_")[0])
    documents = collection.get(ids=[request.document_id])
    if not documents or not documents["documents"]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document_text = documents["documents"][0]
    summary = summarizer.summarize(document_text)
    
    return {"summary": summary, "document_id": request.document_id}

@app.post("/ask")
async def ask_question(request: AskRequest):
    # Retrieve relevant chunks using vector search
    collection = vector_db.get_collection(request.document_id)
    relevant_chunks = collection.search(query=request.query, top_k=3)
    context = " ".join(relevant_chunks)
    # Pass context and query to LLM
    answer = summarizer.summarize(f"Context: {context}\n\nQuestion: {request.query}")
    return {"answer": answer}

#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=8000)