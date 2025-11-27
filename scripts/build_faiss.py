"""Build a FAISS index from `data/chunks.jsonl` using Google embeddings.

Best practices applied:
- Google GenAI `text-embedding-004` (or override via GOOGLE_EMBEDDING_MODEL) is
    required so document and query vectors live in the same space.
- Vectors are L2-normalized and stored in an `IndexFlatIP`, which effectively
    yields cosine similarity search.
- Embedding metadata (provider/model/normalization) is written to
    `data/faiss/embedding_config.json` so the server can recreate the exact
    embedder for queries.

Run (Windows PowerShell):
    .\myenv\Scripts\Activate.ps1
    $Env:GOOGLE_API_KEY="..."
    python scripts\build_faiss.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - build step
    raise RuntimeError("faiss-cpu is required to build the vector store") from exc

from rag.embedding_provider import get_embedding_provider, save_embedding_config

DATA_DIR = ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
FAISS_DIR = DATA_DIR / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")


def load_chunks(path: Path) -> List[dict]:
    if not path.exists():
        print(f"Chunks file not found: {path}. Run scripts/ingest_books.py first.")
        sys.exit(1)

    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            docs.append(json.loads(line))
    return docs


def build_faiss_index(docs: List[dict]):
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    provider = get_embedding_provider(require_google=True)
    print(f"Using embeddings provider {provider.provider} / {provider.model}")

    embeddings = provider.embed_documents(texts)
    emb_matrix = np.array(embeddings, dtype="float32")

    if provider.normalize_l2:
        faiss.normalize_L2(emb_matrix)

    dim = emb_matrix.shape[1]
    if provider.similarity_metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(emb_matrix)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    meta_file = FAISS_DIR / "metadata.jsonl"
    with open(meta_file, "w", encoding="utf-8") as mfh:
        for doc, doc_id in zip(docs, ids):
            meta = {
                "id": doc_id,
                "source": doc.get("source"),
                "chunk_index": doc.get("chunk_index"),
                "page_index_start": doc.get("page_index_start"),
                "page_index_end": doc.get("page_index_end"),
            }
            mfh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    save_embedding_config(FAISS_DIR / "embedding_config.json", provider.as_config())
    print(f"FAISS index + metadata saved to {FAISS_DIR}. Embedding config recorded.")


def main():
    docs = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(docs)} chunks")
    build_faiss_index(docs)


if __name__ == "__main__":
    main()
