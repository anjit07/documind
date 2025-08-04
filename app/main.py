from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from typing import List
import uuid

from app.utils.pdf_processor import PDFProcessor
from app.utils.chuck_file import ChunkFile
from app.vector_storage.vector_db import VectorDB
from app.models.schemas import PDFUpload,SummaryRequest, SummaryResponse, AskRequest
from app.configuration.config import settings
from app.services.summarizer import Summarizer
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
chunkFile = ChunkFile()
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
    
    # process PDF file and extract text and return documents
    documents = pdf_processor.extract_text(file_path)
    if not documents:
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF")

    # Chunk the documents
    chunks = chunkFile.recursive_chunking(documents)

    # Generate unique IDs for each document
    doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    # Store in vector DB
    collection_name = filename.split(".")[0]
    savedData =  vector_db.create_collection(collection_name, chunks, doc_ids)

    print("savedData :", savedData._collection.count())
    print("Collection name:", savedData._collection.name)

    # To fetch and print all document IDs:
    #print("Document IDs:", savedData._collection.get()["ids"])

    return doc_ids

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummaryRequest):
   

    collection = vector_db.get_collection(request.document_name)
    print("summarize collection :", collection._collection.count())
    document_text = [doc for doc in collection.get()["documents"]]
    #print("document_text :", document_text)
    summary = summarizer.summarize(document_text)
    print("summary :", summary)

    return {"summary": summary, "document_name": request.document_name}

@app.post("/ask", response_model=SummaryResponse)
async def ask_question(request: AskRequest):

    # RAG 

    # Retrieval : Retrieve relevant chunks using vector search
    collection = vector_db.get_collection(request.document_name)
    print("ask collection :", collection._collection.count())

    # Augmentation : Extract text content from the Document objects
    document_text = [doc for doc in collection.get()["documents"]]

    # Generation :LLM uses the context + query to generate an answer
    answer = summarizer.ask(document_text, request.query)
    
    print("answer :", answer)

    return {"summary": answer, "document_name": request.document_name}


@app.post("/chartwith", response_model=SummaryResponse)
async def chartwith_question(request: AskRequest):

    # Retrieval : Retrieve relevant chunks using vector search
    collection = vector_db.get_collection(request.document_name)
    print("chartwith collection :", collection._collection.count())
    search_data=  collection.similarity_search(request.query)

    # Augmentation : Extract text content from the Document objects
    retrieved_texts = [doc.page_content for doc in search_data]
    # Format them into a single context string
    context = "\n\n".join(retrieved_texts)
    

    # Generation :LLM uses the context + query to generate an answer.
    answer = summarizer.chart_with(context, request.query)

    print("chartwith_question Answer :", answer)

    return {"summary": answer, "document_name": request.document_name}


    return {"summary": answer, "document_name": request.document_name}

@app.post("/query", response_model=SummaryResponse)
async def query_document(request: AskRequest):

    collection = vector_db.get_collection(request.document_name)
    print("query_document  count :", collection._collection.count())
    search_data=  vector_db.search(request.document_name, request.query)

    # Step 1: Extract text content from the Document objects
    retrieved_texts = [doc.page_content for doc in search_data]
    # Step 2: Format them into a single context string
    context = "\n\n".join(retrieved_texts)

    print("context## :", context)
    answer =summarizer.ask(context, request.query)

    print("query_document Answer :", answer)

    return {"summary": answer, "document_name": request.document_name}



#if __name__ == "__main__":
 #   import uvicorn
  #  uvicorn.run(app, host="0.0.0.0", port=8000)