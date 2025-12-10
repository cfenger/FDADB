import logging
import mimetypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .models import (
    Citation,
    DeleteAllResponse,
    DocumentOut,
    DocumentsResponse,
    ErrorResponse,
    QARequest,
    QAResponse,
    UploadResponse,
)
from .rag_service import RAGService
from .storage import MetadataStore


logger = logging.getLogger(__name__)


@lru_cache
def _get_metadata_store() -> MetadataStore:
    settings = get_settings()
    return MetadataStore(settings.metadata_path)


@lru_cache
def _get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(settings, _get_metadata_store())


settings = get_settings()
app = FastAPI(title="FDA Document QA (RAG)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(static_dir / "index.html")


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@app.get("/api/documents", response_model=DocumentsResponse, responses={500: {"model": ErrorResponse}})
def list_documents(metadata_store: MetadataStore = Depends(_get_metadata_store)):
    docs: list[DocumentOut] = []
    for doc in metadata_store.list_documents():
        payload = doc.model_dump(exclude={"local_path"})
        payload["viewable"] = bool(doc.local_path and Path(doc.local_path).exists())
        docs.append(DocumentOut.model_validate(payload))
    return DocumentsResponse(documents=docs)


@app.get(
    "/api/documents/{document_id}/content",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_document_content(
    document_id: str,
    metadata_store: MetadataStore = Depends(_get_metadata_store),
    rag: RAGService = Depends(_get_rag_service),
):
    doc = metadata_store.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    # Only serve if we have a local stored copy
    if not doc.local_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document content is unavailable locally. Please re-upload to enable viewing.",
        )
    file_path = Path(doc.local_path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document content is unavailable locally. Please re-upload to enable viewing.",
        )

    mime_type = doc.mime_type or mimetypes.guess_type(doc.original_filename)[0] or "application/octet-stream"
    safe_name = "".join(ch for ch in doc.original_filename if 32 <= ord(ch) < 127 and ch not in {'"', "\\"})
    safe_name = safe_name or f"document-{doc.id}"
    headers = {"Content-Disposition": f'inline; filename="{safe_name}"'}
    return FileResponse(
        file_path,
        media_type=mime_type,
        headers=headers,
    )


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def upload_files(
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    rag: RAGService = Depends(_get_rag_service),
):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")
    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files in one request. Max allowed is {settings.max_files_per_request}.",
        )

    prepared: List[Tuple[str, Path, str | None]] = []
    records: List[DocumentOut] = []
    try:
        for upload in files:
            ext = Path(upload.filename).suffix.lower()
            if ext not in settings.allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type '{ext}' is not allowed",
                )

            content = await upload.read()
            if len(content) > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{upload.filename}' exceeds {settings.max_file_size_mb} MB limit",
                )

            temp_path = settings.uploads_dir / f"{uuid4()}{ext}"
            settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            with temp_path.open("wb") as f:
                f.write(content)

            prepared.append((upload.filename, temp_path, upload.content_type))

        raw_records = await run_in_threadpool(rag.upload_files, prepared)
        for rec in raw_records:
            payload = rec.model_dump(exclude={"local_path"})
            payload["viewable"] = bool(rec.local_path and Path(rec.local_path).exists())
            records.append(DocumentOut.model_validate(payload))
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed due to an unexpected error",
        ) from exc
    finally:
        for _, path, _ in prepared:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
    successful_statuses = {"ready", "indexing", "skipped_duplicate", "retry_suggested"}
    successful = [r for r in records if r.processing_status in successful_statuses]
    if not successful:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="All files failed to upload",
        )
    return UploadResponse(documents=records)


@app.delete(
    "/api/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def delete_document(
    document_id: str,
    metadata_store: MetadataStore = Depends(_get_metadata_store),
    rag: RAGService = Depends(_get_rag_service),
):
    doc = metadata_store.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    try:
        rag.delete_document(doc)
        metadata_store.delete_document(document_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.delete(
    "/api/documents",
    response_model=DeleteAllResponse,
    responses={500: {"model": ErrorResponse}},
)
def delete_all_documents(
    metadata_store: MetadataStore = Depends(_get_metadata_store),
    rag: RAGService = Depends(_get_rag_service),
):
    docs = metadata_store.list_documents()
    total = len(docs)
    if not docs:
        return DeleteAllResponse(deleted=0)

    def _background_delete(records):
        try:
            rag.delete_documents(records)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background delete_all_documents failed: %s", exc)

    # Fire-and-forget deletion so the HTTP request doesn't hang on slow OpenAI calls.
    thread = threading.Thread(target=_background_delete, args=(docs,), daemon=True)
    thread.start()

    metadata_store.clear_documents()
    return DeleteAllResponse(deleted=total)


@app.post(
    "/api/ask",
    response_model=QAResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ask_question(
    payload: QARequest,
    rag: RAGService = Depends(_get_rag_service),
    metadata_store: MetadataStore = Depends(_get_metadata_store),
):
    if not payload.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    if not metadata_store.list_documents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents uploaded yet. Please upload files before asking questions.",
        )

    answer_text, sources, citations = await run_in_threadpool(rag.ask, payload.question.strip())
    source_docs = []
    for doc in sources:
        payload = doc.model_dump(exclude={"local_path"})
        payload["viewable"] = bool(doc.local_path and Path(doc.local_path).exists())
        source_docs.append(DocumentOut.model_validate(payload))

    citation_items: list[Citation] = []
    for item in citations:
        snippet = item.get("snippet") or "Snippet unavailable."
        index = item.get("citation_index") or (len(citation_items) + 1)
        citation_items.append(
            Citation(
                citation_index=index,
                original_filename=item.get("original_filename") or "Unknown source",
                openai_file_id=item.get("openai_file_id"),
                snippet=snippet,
            )
        )

    return QAResponse(answer=answer_text, sources=source_docs, citations=citation_items)
