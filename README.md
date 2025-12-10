# FDA Document Q&A (RAG)

Python FastAPI app that lets you upload company documents once and ask grounded questions using the OpenAI Responses API with File Search (OpenAI Python 2.x SDK).

## Features
- Two screens: upload documents and ask questions (single shared environment, no auth).
- Upload multiple files with type/size validation, push to OpenAI file storage, and attach to a single shared vector store.
- Persist minimal metadata (file name, OpenAI file id, vector store id, timestamps) in `data/metadata.json`.
- Q&A endpoint calls the Responses API with `file_search` configured against the shared vector store and returns answers plus source citations.
- Lightweight web UI (vanilla JS) with loading/error states, session-only history, and basic visual polish.
- Delete individual documents or clear all documents (removes OpenAI file + detaches from the shared vector store, then drops local metadata).
- View the original uploaded file contents directly from the document list.
- Uploads are deduplicated by filename (case-insensitive); duplicate filenames in later uploads are skipped automatically while the rest of the batch continues.
- Upload robustness: configurable indexing timeout, per-file statuses (`ready`, `indexing`, `failed`, `retry_suggested`, `skipped_duplicate`), retry/backoff for transient OpenAI errors, and client-side throttled uploads with a retry button for failed files.

## Quickstart
1) Install dependencies (Python 3.10+ recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
# Confirm the OpenAI SDK is 2.x:
# python - <<'PY'
# import openai; print(openai.__version__)
# PY
```

2) Configure environment:
- Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and optional settings (model, size limits, etc.).
- Minimum required entries:
```
OPENAI_API_KEY=sk-...
FDA_QA_MODEL=gpt-5.1
# Optional: tune FILE_INDEX_TIMEOUT_SECONDS to control how long we wait for OpenAI indexing before returning.
```

3) Run the app:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 to use the UI.

## API
- `GET /api/documents` - list known documents.
- `GET /api/documents/{document_id}/content` - download/view the stored file for a document.
- `POST /api/upload` (multipart, `files`) - uploads files, stores them in the shared vector store, and records metadata.
- `POST /api/ask` (JSON `{"question": "..."}`) - runs Responses API with File Search and returns `answer` plus `sources`.
- `DELETE /api/documents/{document_id}` - remove a single document (OpenAI file + vector store entry + metadata).
- `DELETE /api/documents` - remove all documents.
- `GET /health` - simple health probe.

Example curl:
```bash
curl -F "files=@samples/benefits_overview.txt" http://localhost:8000/api/upload
curl -H "Content-Type: application/json" -d "{\"question\": \"What is the PTO policy?\"}" http://localhost:8000/api/ask
```

## Architecture Notes
- `app/config.py`: Pydantic settings reading `.env` (model name, file limits, CORS, paths).
- `app/storage.py`: JSON-backed metadata store with a thread lock, tracking OpenAI file ids and timestamps.
- `app/rag_service.py`: OpenAI client wrapper; creates the shared vector store, uploads files, waits for embedding completion, and runs responses with `file_search` (using the OpenAI Python 2.x vector store endpoints).
- `app/main.py`: FastAPI app wiring dependencies, upload/Q&A endpoints, and static UI serving.
- Frontend at `static/index.html`: vanilla JS for uploads, question submission, loading/error states, and session history.

## Demo Tips
- Pre-load sample docs from `samples/` via the upload page.
- Show that uploads appear in "Known documents", then switch to "Ask questions" and query specifics to surface citations.

## Safeguards and Limits
- Allowed file types default: `.txt, .md, .pdf, .docx`; size limit default: 10 MB per file (configurable via `.env`).
- All users share a single vector store (no per-user isolation in this MVP).
- Metadata is stored locally in `data/metadata.json`; vector store id is reused across runs.
- Large files can take time to index. The app may report "uploaded, indexing in progress" while OpenAI finishes processing. Adjust `FILE_INDEX_TIMEOUT_SECONDS` in `.env` to change how long the backend waits before returning.
- Uploads return per-file statuses (`ready`, `indexing`, `failed`, `retry_suggested`, `skipped_duplicate`). One slow/failed/duplicate file in a batch does not block other files.
- Deleting documents removes their OpenAI file and detaches them from the shared vector store; metadata is updated accordingly. The vector store id is retained for reuse.
- Duplicate uploads are skipped by filename (case-insensitive) and reported in the upload response, but they are not stored again.
- Recommended upload limits: defaults are max 10 files per request (`MAX_FILES_PER_REQUEST`) and 10 MB per file (`MAX_FILE_SIZE_MB`). Uploads are throttled on the client to avoid OpenAI rate limits, with exponential backoff on transient OpenAI errors (429/5xx/timeouts).
- On timeouts, files are marked `indexing` with a timeout message; on transient errors, files are marked `retry_suggested` so you can re-run them via the "Retry failed uploads" button. Local validation errors (size/type) return 400 with a clear message.
