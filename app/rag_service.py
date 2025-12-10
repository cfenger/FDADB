import logging
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI, RateLimitError

from .config import Settings
from .storage import DocumentRecord, MetadataStore


logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings, metadata_store: MetadataStore):
        self.settings = settings
        self.metadata_store = metadata_store
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.vector_store_id = self._ensure_vector_store()
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # Vector store setup -------------------------------------------------
    def _ensure_vector_store(self) -> str:
        existing = self.metadata_store.vector_store_id
        if existing:
            return existing
        vector_store = self.client.vector_stores.create(name=self.settings.vector_store_name)
        self.metadata_store.set_vector_store_id(vector_store.id)
        return vector_store.id

    # Upload flow --------------------------------------------------------
    def upload_files(self, uploads: Iterable[Tuple[str, Path, Optional[str]]]) -> List[DocumentRecord]:
        """
        uploads: iterable of tuples (original_filename, path_on_disk, mime_type)
        """
        records: List[DocumentRecord] = []
        for original_filename, path, mime_type in uploads:
            if self.metadata_store.exists_by_filename(original_filename):
                logger.info(
                    "Skipping duplicate upload for filename=%s (already exists in metadata)", original_filename
                )
                records.append(
                    DocumentRecord(
                        id=str(uuid4()),
                        original_filename=original_filename,
                        openai_file_id=None,
                        vector_store_id=self.vector_store_id,
                        size_bytes=path.stat().st_size,
                        mime_type=mime_type,
                        processing_status="skipped_duplicate",
                        error_message="Skipped duplicate filename (already uploaded).",
                    )
                )
                continue

            processing_status = "ready"
            openai_file_id: Optional[str] = None
            error_message: Optional[str] = None
            try:
                with path.open("rb") as f:
                    openai_file = self._execute_with_retries(
                        lambda: self.client.files.create(file=f, purpose="assistants"),
                        "upload file",
                    )
                openai_file_id = getattr(openai_file, "id", None)

                try:
                    self._execute_with_retries(
                        lambda: self.client.vector_stores.files.create(
                            vector_store_id=self.vector_store_id, file_id=openai_file_id
                        ),
                        "attach file to vector store",
                    )
                    self._wait_for_file_ready(
                        openai_file_id, timeout_seconds=self.settings.file_index_timeout_seconds
                    )
                except TimeoutError as exc:
                    processing_status = "indexing"
                    error_message = f"Indexing timed out after {self.settings.file_index_timeout_seconds}s."
                    logger.warning(
                        "Indexing timed out for file_id=%s filename=%s: %s",
                        openai_file_id or "<unknown>",
                        original_filename,
                        exc,
                    )
                except Exception as exc:  # noqa: BLE001
                    if self._is_transient_error(exc):
                        processing_status = "retry_suggested"
                    else:
                        processing_status = "failed"
                    error_message = self._format_openai_error(exc)
                    logger.exception(
                        "Failed to attach or index file '%s' (file_id=%s)", original_filename, openai_file_id
                    )
            except Exception as exc:  # noqa: BLE001
                if self._is_transient_error(exc):
                    processing_status = "retry_suggested"
                else:
                    processing_status = "failed"
                error_message = self._format_openai_error(exc)
                logger.exception(
                    "Failed to upload file '%s' (status=%s)", original_filename, processing_status
                )

            if processing_status in {"failed", "retry_suggested"} and openai_file_id:
                self._cleanup_openai_file(openai_file_id, f"{processing_status} during upload")
                openai_file_id = None

            record = DocumentRecord(
                id=str(uuid4()),
                original_filename=original_filename,
                openai_file_id=openai_file_id,
                vector_store_id=self.vector_store_id,
                size_bytes=path.stat().st_size,
                mime_type=mime_type,
                processing_status=processing_status,
                error_message=error_message,
            )
            if processing_status not in {"skipped_duplicate", "failed", "retry_suggested"}:
                self.metadata_store.add_document(record)
            records.append(record)
        return records

    def _wait_for_file_ready(self, file_id: str, timeout_seconds: int) -> None:
        start = time.time()
        while True:
            file_status = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=file_id
            )
            status = getattr(file_status, "status", None) or getattr(file_status, "state", None)
            if status is None and isinstance(file_status, dict):
                status = file_status.get("status") or file_status.get("state")

            normalized = status.lower() if isinstance(status, str) else ""
            if normalized in {"completed", "ready", "succeeded"}:
                return
            if normalized in {"failed", "cancelled", "error"}:
                raise RuntimeError(f"File {file_id} failed to process: status={status}")
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timeout while processing file {file_id}")
            time.sleep(2)

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {429, 500, 502, 503}
        return False

    def _execute_with_retries(
        self,
        func: Callable[[], object],
        description: str,
        *,
        max_retries: int = 2,
        base_delay: float = 1.5,
    ):
        attempt = 0
        while True:
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if not self._is_transient_error(exc) or attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error during %s (attempt %s/%s): %s",
                    description,
                    attempt,
                    max_retries,
                    self._format_openai_error(exc),
                )
                time.sleep(delay)

    @staticmethod
    def _format_openai_error(exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        err_type = getattr(exc, "type", None)
        message = getattr(exc, "message", None) or str(exc)
        parts = []
        if status:
            parts.append(f"status={status}")
        if err_type:
            parts.append(f"type={err_type}")
        if code:
            parts.append(f"code={code}")
        parts.append(f"message={message}")
        return "; ".join(parts)

    def _cleanup_openai_file(self, file_id: str, reason: str) -> None:
        try:
            self.client.files.delete(file_id)
            logger.info("Deleted orphaned OpenAI file %s (%s)", file_id, reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to delete orphaned OpenAI file %s (%s): %s", file_id, reason, exc, exc_info=False
            )

    # Q&A flow -----------------------------------------------------------
    def delete_document(self, record: DocumentRecord) -> None:
        """
        Removes a document's OpenAI resources (detach from vector store, delete file).
        """
        openai_file_id = record.openai_file_id
        if not openai_file_id:
            return
        vector_store_id = record.vector_store_id or self.vector_store_id
        if vector_store_id:
            try:
                self.client.vector_stores.files.delete(
                    file_id=openai_file_id,
                    vector_store_id=vector_store_id,
                )
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "status_code", None) == 404:
                    logger.warning(
                        "Vector store file already missing (file_id=%s, vector_store_id=%s)",
                        openai_file_id,
                        vector_store_id,
                    )
                else:
                    logger.exception(
                        "Failed detaching file_id=%s from vector_store=%s",
                        openai_file_id,
                        vector_store_id,
                    )
                    raise
        try:
            self.client.files.delete(openai_file_id)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "status_code", None) == 404:
                logger.warning("OpenAI file already missing (file_id=%s)", openai_file_id)
            else:
                logger.exception(
                    "Failed to delete OpenAI file '%s' (file_id=%s)", record.original_filename, openai_file_id
                )
                raise

    def delete_documents(self, records: Iterable[DocumentRecord]) -> None:
        for record in records:
            self.delete_document(record)

    def ask(self, question: str) -> Tuple[str, List[DocumentRecord]]:
        response = self.client.responses.create(
            model=self.settings.qa_model,
            input=[{"role": "user", "content": question}],
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
        )

        answer_text = self._extract_output_text(response)
        file_ids = self._extract_citation_ids(response)
        sources: List[DocumentRecord] = []
        for file_id in file_ids:
            doc = self.metadata_store.get_by_file_id(file_id)
            if doc:
                sources.append(doc)
        return answer_text, sources

    @staticmethod
    def _extract_output_text(response) -> str:
        # The Responses API exposes a convenience attribute; fall back to manual extraction.
        text = getattr(response, "output_text", None)
        if text:
            return text

        output = getattr(response, "output", None) or []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", None) or []
            for block in content:
                if isinstance(block, dict):
                    text_obj = block.get("text", {}) or {}
                else:
                    text_obj = getattr(block, "text", None) or {}
                if isinstance(text_obj, dict):
                    value = text_obj.get("value")
                else:
                    value = getattr(text_obj, "value", None)
                if value:
                    return value
        return ""

    @staticmethod
    def _extract_citation_ids(response) -> Set[str]:
        ids: Set[str] = set()
        output = getattr(response, "output", None) or []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", None) or []
            for block in content:
                if isinstance(block, dict):
                    text_obj = block.get("text", {}) or {}
                else:
                    text_obj = getattr(block, "text", None) or {}

                if isinstance(text_obj, dict):
                    annotations = text_obj.get("annotations", []) or []
                else:
                    annotations = getattr(text_obj, "annotations", None) or []

                for ann in annotations:
                    if isinstance(ann, dict):
                        ann_type = ann.get("type")
                        file_citation = ann.get("file_citation", {}) or {}
                        file_id = file_citation.get("file_id")
                    else:
                        ann_type = getattr(ann, "type", None)
                        file_citation = getattr(ann, "file_citation", None)
                        file_id = getattr(file_citation, "file_id", None) if file_citation else None
                    if ann_type != "file_citation":
                        continue
                    if file_id:
                        ids.add(file_id)
        return ids
