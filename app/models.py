from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    original_filename: str
    openai_file_id: Optional[str] = None
    vector_store_id: Optional[str] = None
    uploaded_at: datetime
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    processing_status: str = Field(default="ready")
    error_message: Optional[str] = None


class UploadResponse(BaseModel):
    documents: List[DocumentOut]


class DocumentsResponse(BaseModel):
    documents: List[DocumentOut]


class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class QAResponse(BaseModel):
    answer: str
    sources: List[DocumentOut]


class ErrorResponse(BaseModel):
    detail: str


class DeleteAllResponse(BaseModel):
    deleted: int
