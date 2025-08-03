from pydantic import BaseModel

class PDFUpload(BaseModel):
    filename: str

class SummaryRequest(BaseModel):
    document_name: str

class SummaryResponse(BaseModel):
    summary: str
    document_name: str

class AskRequest(BaseModel):
    document_name: str
    query: str