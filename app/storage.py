import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    original_filename: str
    openai_file_id: Optional[str] = None
    vector_store_id: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    processing_status: str = Field(default="ready")
    error_message: Optional[str] = None


class Metadata(BaseModel):
    vector_store_id: Optional[str] = None
    documents: List[DocumentRecord] = Field(default_factory=list)


class MetadataStore:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Metadata:
        if not self._path.exists():
            return Metadata()
        with self._path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return Metadata(**raw)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data.model_dump(mode="json"), f, indent=2)

    @property
    def vector_store_id(self) -> Optional[str]:
        with self._lock:
            return self._data.vector_store_id

    def set_vector_store_id(self, vector_store_id: str) -> None:
        with self._lock:
            if self._data.vector_store_id != vector_store_id:
                self._data.vector_store_id = vector_store_id
                self._save()

    def add_document(self, record: DocumentRecord) -> None:
        with self._lock:
            self._data.documents.append(record)
            self._save()

    def exists_by_filename(self, original_filename: str) -> bool:
        normalized = original_filename.lower()
        with self._lock:
            return any(
                doc.original_filename.lower() == normalized
                and doc.processing_status not in {"failed", "retry_suggested"}
                for doc in self._data.documents
            )

    def list_documents(self) -> List[DocumentRecord]:
        with self._lock:
            # return a copy to avoid external mutation
            return list(self._data.documents)

    def get_by_file_id(self, file_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            for doc in self._data.documents:
                if doc.openai_file_id == file_id:
                    return doc
            return None

    def get_by_id(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            for doc in self._data.documents:
                if doc.id == doc_id:
                    return doc
            return None

    def delete_document(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            remaining: List[DocumentRecord] = []
            removed: Optional[DocumentRecord] = None
            for doc in self._data.documents:
                if doc.id == doc_id:
                    removed = doc
                else:
                    remaining.append(doc)
            if removed:
                self._data.documents = remaining
                self._save()
            return removed

    def clear_documents(self) -> List[DocumentRecord]:
        with self._lock:
            removed = list(self._data.documents)
            self._data.documents = []
            self._save()
            return removed
