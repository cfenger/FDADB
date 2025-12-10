from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_index: int
    original_filename: str
    openai_file_id: Optional[str] = None
    snippet: Optional[str] = None
    document_id: Optional[str] = None
    viewable: bool = False


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
    viewable: bool = Field(default=False, description="True if a local copy is available to stream")


class UploadResponse(BaseModel):
    documents: List[DocumentOut]


class DocumentsResponse(BaseModel):
    documents: List[DocumentOut]


class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class QAResponse(BaseModel):
    answer: str
    sources: List[DocumentOut]
    citations: List[Citation] = Field(
        default_factory=list,
        description="Per-answer citation details in the order they appear in the answer",
    )


class ErrorResponse(BaseModel):
    detail: str


class DeleteAllResponse(BaseModel):
    deleted: int
