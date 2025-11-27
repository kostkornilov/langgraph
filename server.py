"""Minimal FastAPI server for RAG chat over the FAISS index.

Endpoints:
- GET / -> serves `web/index.html`
- POST /api/search -> {"query": ..., "top_k": 3} returns top_k chunks from FAISS
- POST /api/chat -> {"query": ..., "top_k": 3, "thread_id": "1", "user_id": "1"}
    If an LLM model is configured (env MODEL_ID), the server will attempt to call it via LangChain.

Run (PowerShell):
  .\myenv\Scripts\Activate.ps1
  pip install -r requirements.txt
  uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import server_conversations
from dotenv import load_dotenv

from rag.embedding_provider import (
    get_embedding_provider,
    load_embedding_config,
)

ROOT = Path(__file__).resolve().parent
DATA_FAISS = ROOT / "data" / "faiss"

# Load .env if present so local credentials (GOOGLE_API_KEY, MODEL_ID, etc.) are available
load_dotenv(ROOT / ".env")

app = FastAPI(title="LangGraph RAG Server")


def load_faiss_store():
    try:
        import faiss
        import numpy as np
    except Exception as e:  # pragma: no cover - runtime guard
        raise RuntimeError("faiss-cpu and numpy are required on the server") from e

    index_path = DATA_FAISS / "index.faiss"
    meta_path = DATA_FAISS / "metadata.jsonl"
    config_path = DATA_FAISS / "embedding_config.json"

    if not index_path.exists() or not meta_path.exists() or not config_path.exists():
        raise FileNotFoundError("FAISS index, metadata, or embedding_config.json missing in data/faiss/")

    config = load_embedding_config(config_path)
    provider = get_embedding_provider(expected_provider=config.provider, expected_model=config.model)

    idx = faiss.read_index(str(index_path))

    # load metadata lines (order should match FAISS vectors)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            metas.append(json.loads(line))

    # Also load chunk texts so we can return excerpts
    chunks_path = ROOT / "data" / "chunks.jsonl"
    id_to_text: Dict[str, str] = {}
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as chf:
            for line in chf:
                try:
                    doc = json.loads(line)
                    id_to_text[doc.get("id")] = doc.get("text", "")
                except Exception:
                    continue

    def embed_query(text: str):
        vec = np.array(provider.embed_query(text), dtype="float32")
        if config.normalize_l2:
            faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    class FaissWrapper:
        def __init__(self, index, metas):
            self.index = index
            self.metas = metas

        def similarity_search(self, query: str, k: int = 3):
            qv = embed_query(query).reshape(1, -1)
            D, I = self.index.search(qv, k)
            results = []
            for idx_pos in I[0]:
                if idx_pos < 0 or idx_pos >= len(self.metas):
                    continue
                meta = self.metas[idx_pos]

                class Doc:
                    def __init__(self, meta, text):
                        self.metadata = meta
                        self.page_content = text

                text = id_to_text.get(meta.get("id"), "")
                results.append(Doc(meta, text))
            return results

    return FaissWrapper(idx, metas)


def try_init_model():
    # Try to initialize a LangChain chat model if environment variable MODEL_ID is set
    model_id = os.environ.get("MODEL_ID") or os.environ.get("MODEL")
    # If MODEL_ID isn't set but a Google API key is present, attempt a sensible default
    if not model_id and os.environ.get("GOOGLE_API_KEY"):
        # allow overriding with GOOGLE_MODEL_ID if the user set it
        model_id = os.environ.get("GOOGLE_MODEL_ID") or os.environ.get("GOOGLE_MODEL")
        if not model_id:
            # Conservative default — adjust to your available GenAI offering
            model_id = "google_genai:gemini-2.5-flash-lite"
        print(f"No MODEL_ID set; falling back to Google model_id={model_id} because GOOGLE_API_KEY is present")
    if not model_id:
        return None
    try:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model_id, temperature=0.0, timeout=10)
        return model
    except Exception as e:
        # Return None if model can't be initialized; server will still respond with chunks
        print(f"Model init failed: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = ROOT / "web" / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return FileResponse(index_file)


@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    # Prevent path traversal
    safe_name = Path(filename).name
    pdf_path = ROOT / "books" / safe_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path)


@app.get("/health")
async def health():
    """Liveness endpoint: returns 200 if server is up."""
    return JSONResponse({"status": "ok"})


@app.get("/ready")
async def ready():
    """Readiness endpoint: attempts to load FAISS store and reports status.

    This is intentionally lightweight: it will try to import/load minimal components
    to verify the vector store is accessible. It returns 200 when ready and 503 when not.
    """
    try:
        # Try to load faiss store; this may be slow if model downloads occur, but
        # it's useful for readiness checks in containers where data/faiss is mounted.
        _ = load_faiss_store()
        return JSONResponse({"ready": True})
    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)}, status_code=503)


@app.post("/api/search")
async def api_search(req: Request):
    body = await req.json()
    query = body.get("query")
    top_k = int(body.get("top_k", 3))
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    try:
        store = load_faiss_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        results = store.similarity_search(query, k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"similarity_search failed: {e}")

    # Each result is a Document-like object; try to serialize common fields
    out = []
    for d in results:
        meta = getattr(d, "metadata", {}) or {}
        page_start = meta.get("page_index_start")
        source = meta.get("source")
        link = None
        if source is not None and page_start is not None:
            # PDF viewers expect 1-indexed page numbers
            link = f"/pdf/{source}#page={int(page_start) + 1}"

        out.append({
            "page_index_start": page_start,
            "page_index_end": meta.get("page_index_end"),
            "source": source,
            "chunk_index": meta.get("chunk_index"),
            "text": getattr(d, "page_content", str(d))[:2000],
            "link": link,
        })

    return JSONResponse({"query": query, "results": out})


@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    query = body.get("query")
    top_k = int(body.get("top_k", 3))
    thread_id = body.get("thread_id", "1")
    user_id = body.get("user_id", "1")

    if not query:
        raise HTTPException(status_code=400, detail="query required")

    try:
        store = load_faiss_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        docs = store.similarity_search(query, k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"similarity_search failed: {e}")

    # Prepare context for LLM
    context_parts = []
    citations = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source")
        chunk_index = meta.get("chunk_index")
        text = getattr(d, "page_content", str(d))
        # create link to PDF page (if available)
        page_start = meta.get("page_index_start")
        link = None
        if source is not None and page_start is not None:
            link = f"/pdf/{source}#page={int(page_start) + 1}"

        # include a truncated excerpt for display in the UI
        excerpt = text if not text else (text[:2000] + ("..." if len(text) > 2000 else ""))

        citations.append({"source": source, "chunk_index": chunk_index, "link": link, "text": excerpt})
        context_parts.append(f"SOURCE: {source} CHUNK: {chunk_index}\n{text}")

    prompt = "\n\n".join(context_parts)

    model = try_init_model()
    if not model:
        # Return retrieved context and a suggested prompt if LLM isn't configured
        # persist user message
        try:
            server_conversations.save_message(thread_id, "user", query)
        except Exception:
            pass
        return JSONResponse({"query": query, "prompt": prompt, "citations": citations, "note": "LLM not configured (set MODEL_ID env). Returned context only."})

    # If model exists, call it with a lightweight prompt
    try:
        # Minimal prompt that asks for an answer and to include citations
        system = {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided SOURCES to answer the user and include citation metadata (source + chunk_index)."
        }
        user_msg = {"role": "user", "content": f"CONTEXT:\n{prompt}\n\nQuestion: {query}"}

        # Some LangChain chat model wrappers expect a string prompt instead of a dict
        prompt_for_model = f"{system['content']}\n\nCONTEXT:\n{prompt}\n\nQuestion: {query}"
        try:
            response = model.invoke(prompt_for_model, config={"configurable": {"thread_id": thread_id}})
        except TypeError:
            # fallback: some implementations expect a dict with 'input'
            response = model.invoke({"input": prompt_for_model}, config={"configurable": {"thread_id": thread_id}})

        # Normalize response: it may be a string, dict or structured object
        text_out = None
        if isinstance(response, str):
            text_out = response
        elif isinstance(response, dict):
            structured = response.get("structured_response")
            if structured:
                text_out = str(structured)
            else:
                messages = response.get("messages")
                if messages and isinstance(messages, list) and len(messages) > 0:
                    # try last message content
                    last = messages[-1]
                    text_out = last.get("content") if isinstance(last, dict) else str(last)
                else:
                    # fallback to stringified dict
                    text_out = json.dumps(response)
        else:
            # fallback to string conversion
            try:
                text_out = str(response)
            except Exception:
                text_out = ""

        # persist assistant answer
        try:
            server_conversations.save_message(thread_id, "assistant", text_out)
        except Exception:
            pass

        # Return answer plus enriched citations (include excerpt and links for UI)
        return JSONResponse({"answer": text_out, "citations": citations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")
