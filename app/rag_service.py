import logging
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple
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
            record_id = str(uuid4())
            if self.metadata_store.exists_by_filename(original_filename):
                logger.info(
                    "Skipping duplicate upload for filename=%s (already exists in metadata)", original_filename
                )
                records.append(
                    DocumentRecord(
                        id=record_id,
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
            local_path: Optional[str] = None
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

            if processing_status not in {"skipped_duplicate", "failed", "retry_suggested"}:
                local_path = self._persist_local_copy(path, record_id, original_filename)

            record = DocumentRecord(
                id=record_id,
                original_filename=original_filename,
                openai_file_id=openai_file_id,
                vector_store_id=self.vector_store_id,
                size_bytes=path.stat().st_size,
                mime_type=mime_type,
                processing_status=processing_status,
                error_message=error_message,
                local_path=local_path,
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

    def _delete_local_copy(self, record: DocumentRecord) -> None:
        if not record.local_path:
            return
        path = Path(record.local_path)
        try:
            if path.exists():
                path.unlink()
            # clean up empty parent dirs (record-specific)
            if path.parent.exists():
                try:
                    path.parent.rmdir()
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete local copy for %s", record.id, exc_info=False)

    # Q&A flow -----------------------------------------------------------
    def stream_file_content(self, record: DocumentRecord) -> Tuple[Iterator[bytes], str]:
        """
        Stream a document's content from OpenAI Files.
        Returns: (byte iterator, mime_type)
        """
        if not record.openai_file_id:
            raise ValueError("Document has no associated OpenAI file")
        response = self.client.files.content(record.openai_file_id)

        def iterator():
            try:
                yield from response.iter_bytes()
            finally:
                try:
                    response.close()
                except Exception:
                    pass

        mime_type = record.mime_type or mimetypes.guess_type(record.original_filename)[0] or "application/octet-stream"
        return iterator(), mime_type

    def _persist_local_copy(self, src: Path, record_id: str, original_filename: str) -> Optional[str]:
        """
        Store a local copy of the uploaded file so it can be served back to users.
        """
        try:
            safe_name = Path(original_filename).name
            safe_name = "".join(ch for ch in safe_name if ch not in '<>:"/\\|?*') or f"document-{record_id}"
            dest_dir = self.settings.uploads_dir / "stored" / record_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / safe_name
            shutil.copy2(src, dest)
            return str(dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist local copy for %s: %s", original_filename, exc, exc_info=False)
            return None

    def delete_document(self, record: DocumentRecord) -> None:
        """
        Removes a document's OpenAI resources (detach from vector store, delete file).
        """
        openai_file_id = record.openai_file_id
        if not openai_file_id:
            self._delete_local_copy(record)
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
        self._delete_local_copy(record)

    def delete_documents(self, records: Iterable[DocumentRecord]) -> None:
        for record in records:
            self.delete_document(record)

    def ask(self, question: str) -> Tuple[str, List[DocumentRecord], List[dict]]:
        response = self.client.responses.create(
            model=self.settings.qa_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant using the provided FDA documents via file_search. "
                        "Every time you state a fact that comes from one or more documents, you MUST add inline citations like [1], [2] directly after the sentence or phrase they support. "
                        "Do not group citations only at the end of the answer; avoid trailing blocks like [1] [2] [3] on the last line. "
                        "If you do not use any documents, do not add citations."
                    ),
                },
                {"role": "user", "content": question},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [self.vector_store_id]}],
            tool_choice="required",
            include=["file_search_call.results"],
        )
        try:
            logger.info("Responses payload (truncated): %s", response.model_dump(exclude_none=True))
        except Exception:  # noqa: BLE001
            logger.info("Responses payload (fallback str): %s", response)

        # First extract citation details and snippets so we can both
        # build a sources list and map file_ids to citation indices.
        citation_details = self._extract_citations(response)
        snippet_map = self._extract_file_search_snippets(response)

        # If no citations came back, fall back to unique file IDs for sources
        fallback_file_ids = list(self._extract_citation_ids(response))

        citation_payload: List[dict] = []
        ordered_file_ids: List[str] = []

        if citation_details:
            unique_by_file: dict[str, dict] = {}
            for entry in citation_details:
                fid = entry.get("file_id")
                if not fid or fid in unique_by_file:
                    continue
                unique_by_file[fid] = entry
            max_citations = 5
            ordered_file_ids = list(unique_by_file.keys())[:max_citations]
            for idx, fid in enumerate(ordered_file_ids, start=1):
                doc = self.metadata_store.get_by_file_id(fid)
                citation_payload.append(
                    {
                        "citation_index": idx,
                        "original_filename": (doc.original_filename if doc else fid) or "Unknown source",
                        "openai_file_id": fid,
                        "snippet": snippet_map.get(fid),
                    }
                )
        else:
            # No annotations; we still want to return sources if we have them
            ordered_file_ids = fallback_file_ids

        sources: List[DocumentRecord] = []
        for file_id in ordered_file_ids:
            doc = self.metadata_store.get_by_file_id(file_id)
            if doc:
                sources.append(doc)

        # Get the answer text as a single string.
        answer_text = self._extract_output_text(response)

        # If we have citations but no inline markers yet, distribute them
        # heuristically across the answer instead of grouping them only
        # at the very end.
        if citation_payload:
            indices = [c["citation_index"] for c in citation_payload if c.get("citation_index")]
            indices = indices[:5]
            if indices and not any(f"[{i}]" in answer_text for i in indices):
                answer_text = self._distribute_citation_markers(answer_text, indices)

        return answer_text, sources, citation_payload

    @staticmethod
    def _extract_output_text(response) -> str:
        # Prefer the convenience attribute if available.
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            return text

        output = getattr(response, "output", None) or []
        for item in output:
            if isinstance(item, dict):
                content_list = item.get("content", []) or []
            else:
                content_list = getattr(item, "content", None) or []

            for block in content_list:
                # For ResponseOutputText objects, the text is on the
                # `text` attribute. Dict-like fallbacks are kept for
                # robustness.
                if isinstance(block, dict):
                    value = block.get("text")
                else:
                    value = getattr(block, "text", None)
                if isinstance(value, str) and value:
                    return value
        return ""

    @staticmethod
    def _distribute_citation_markers(answer_text: str, indices: List[int]) -> str:
        """
        Heuristically distribute citation markers like [1], [2] across
        the answer text instead of grouping them only at the very end.
        """
        if not answer_text.strip() or not indices:
            return answer_text

        import re

        text = answer_text

        # Find candidate sentence boundaries as insertion points.
        # We treat ".", "?" and "!" as sentence terminators, but try to
        # avoid breaking on decimal points (e.g. "2.5 mg").
        candidate_positions: List[int] = []
        pattern = re.compile(r"(?<!\d)([\.!?])(?!\d)")
        for match in pattern.finditer(text):
            candidate_positions.append(match.end())

        if not candidate_positions:
            # Fall back to appending all markers at the very end.
            tail = " " + " ".join(f"[{i}]" for i in indices)
            return text.rstrip() + tail

        # Spread citation indices roughly evenly over the available
        # candidate positions so markers appear throughout the answer.
        inserts: List[Tuple[int, str]] = []
        n_candidates = len(candidate_positions)
        n_indices = len(indices)
        for idx, citation_index in enumerate(indices, start=1):
            # Position this marker at a proportional point in the text.
            # Example for 3 markers: ~25%, ~50%, ~75% of sentence ends.
            target = int(round(idx * n_candidates / (n_indices + 1))) - 1
            target = max(0, min(n_candidates - 1, target))
            pos = candidate_positions[target]
            inserts.append((pos, f"[{citation_index}]"))

        # Apply inserts from right to left so offsets remain valid.
        inserts.sort(key=lambda t: t[0], reverse=True)
        for pos, marker in inserts:
            if marker in text:
                continue
            if pos < 0 or pos > len(text):
                pos = len(text)
            text = text[:pos] + marker + text[pos:]

        return text

    @staticmethod
    def _extract_file_search_snippets(response) -> dict[str, str]:
        """
        Extract the first snippet per file_id from file_search tool call results.
        """
        snippets: dict[str, str] = {}
        output = getattr(response, "output", None) or []
        for item in output:
            results = None
            if getattr(item, "type", None) == "file_search_call":
                results = getattr(item, "results", None) or []
            elif isinstance(item, dict) and item.get("type") == "file_search_call":
                results = item.get("results", []) or []
            if not results:
                continue
            for res in results:
                fid = getattr(res, "file_id", None) or (res.get("file_id") if isinstance(res, dict) else None)
                text_val = getattr(res, "text", None) or (res.get("text") if isinstance(res, dict) else None)
                if not fid or not text_val or fid in snippets:
                    continue
                snippet = " ".join(text_val.split()).strip()
                if len(snippet) > 200:
                    snippet = snippet[:200].rstrip() + "..."
                snippets[fid] = snippet
        return snippets

    @staticmethod
    def _extract_citations(response) -> List[dict]:
        """
        Return a list of dicts in annotation order (do not deduplicate):
        [{"file_id": "..."}]
        """
        citations: List[dict] = []
        output = getattr(response, "output", None) or []
        for item in output:
            content_list = getattr(item, "content", None) or []
            if isinstance(item, dict):
                content_list = item.get("content", []) or []

            for block in content_list:
                annotations = getattr(block, "annotations", None)

                # Fallback for dict-like structures
                if annotations is None and isinstance(block, dict):
                    text_obj = block.get("text", {}) or {}
                    if isinstance(text_obj, dict):
                        annotations = text_obj.get("annotations", []) or []
                    else:
                        annotations = getattr(text_obj, "annotations", None) or []

                if not annotations:
                    continue

                for ann in annotations:
                    ann_type = getattr(ann, "type", None)
                    if ann_type == "file_citation":
                        fid = getattr(ann, "file_id", None)
                    elif isinstance(ann, dict) and ann.get("type") == "file_citation":
                        file_citation = ann.get("file_citation", {}) or {}
                        fid = file_citation.get("file_id")
                    else:
                        fid = None
                    if fid:
                        citations.append({"file_id": fid})
        return citations

    @staticmethod
    def _extract_citation_ids(response) -> Set[str]:
        ids: Set[str] = set()
        output = getattr(response, "output", None) or []
        for item in output:
            content_list = getattr(item, "content", None) or []
            if isinstance(item, dict):
                content_list = item.get("content", []) or []

            for block in content_list:
                annotations = getattr(block, "annotations", None)
                if annotations is None and isinstance(block, dict):
                    text_obj = block.get("text", {}) or {}
                    if isinstance(text_obj, dict):
                        annotations = text_obj.get("annotations", []) or []
                    else:
                        annotations = getattr(text_obj, "annotations", None) or []

                if not annotations:
                    continue

                for ann in annotations:
                    ann_type = getattr(ann, "type", None)
                    if ann_type == "file_citation":
                        file_id = getattr(ann, "file_id", None)
                    elif isinstance(ann, dict):
                        if ann.get("type") != "file_citation":
                            continue
                        file_citation = ann.get("file_citation", {}) or {}
                        file_id = file_citation.get("file_id")
                    else:
                        file_id = None

                    if file_id:
                        ids.add(file_id)
        return ids
