# Deployment notes

This repo includes a minimal FastAPI server (`server.py`) that serves a static chat UI and exposes `/api/search` and `/api/chat` endpoints.

Dockerfile: a simple container image is provided (`Dockerfile`) which installs Python dependencies, copies the repo, and runs `uvicorn server:app --host 0.0.0.0 --port 8000`.

Environment variables

- `MODEL_ID` or `MODEL` — optional LangChain model id (e.g., `google_genai:gemini-2.5-flash-lite`) used by the server for generation.
- `GOOGLE_API_KEY` — if set, the FAISS build script and server prefer Google GenAI embeddings via `langchain_google_genai`.
- `GOOGLE_EMBEDDING_MODEL` — optional override for the Google embedding model (default: `embed-text-embedding-gecko-001`).

Build considerations

- The `requirements.txt` lists heavy packages (torch, sentence-transformers). For smaller docker images, prefer using cloud embeddings (set `GOOGLE_API_KEY` or `OPENAI_API_KEY`) and remove `sentence-transformers` and `torch` from the image.

Quick run (local)

1. Build and run the Docker image (example):

   ```powershell
   docker build -t langgraph-rag:latest .
   docker run -p 8000:8000 -e GOOGLE_API_KEY="<your-key>" langgraph-rag:latest
   ```

2. The UI will be available at http://localhost:8000/
