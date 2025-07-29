from pydantic import BaseModel

class PDFUpload(BaseModel):
    filename: str

class SummaryRequest(BaseModel):
    document_id: str

class SummaryResponse(BaseModel):
    summary: str
    document_id: str

class AskRequest(BaseModel):
    document_id: str
    query: str