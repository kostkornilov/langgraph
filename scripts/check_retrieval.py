"""Simple verification tool for FAISS retrieval.

Usage (PowerShell):
  .\myenv\Scripts\Activate.ps1
  python .\scripts\check_retrieval.py "Раскольников" --top_k 5
"""
from pathlib import Path
import json
import argparse
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rag.embedding_provider import get_embedding_provider, load_embedding_config

DATA = ROOT / "data"
CHUNKS_FILE = DATA / "chunks.jsonl"
FAISS_DIR = DATA / "faiss"
META_FILE = FAISS_DIR / "metadata.jsonl"
INDEX_FILE = FAISS_DIR / "index.faiss"
CONFIG_FILE = FAISS_DIR / "embedding_config.json"


def load_chunks(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            d[obj["id"]] = obj
    return d


def load_metadata(path):
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def load_faiss_index(index_path):
    try:
        import faiss
    except Exception:
        raise RuntimeError("faiss is not installed in this environment")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    idx = faiss.read_index(str(index_path))
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    if not CHUNKS_FILE.exists():
        print("Chunks file not found:", CHUNKS_FILE)
        return
    if not META_FILE.exists() or not INDEX_FILE.exists():
        print("FAISS metadata or index not found in:", FAISS_DIR)
        return

    if not CONFIG_FILE.exists():
        print("Embedding config missing:", CONFIG_FILE)
        return

    chunks = load_chunks(CHUNKS_FILE)
    meta = load_metadata(META_FILE)
    index = load_faiss_index(INDEX_FILE)
    config = load_embedding_config(CONFIG_FILE)
    provider = get_embedding_provider(expected_provider=config.provider, expected_model=config.model)

    qv = np.array(provider.embed_query(args.query), dtype="float32")
    if config.normalize_l2:
        qv = qv / (np.linalg.norm(qv) + 1e-12)
    qv_arr = np.asarray([qv], dtype="float32")

    D, I = index.search(qv_arr, args.top_k)

    print(f"Query: {args.query!r}  (top_k={args.top_k})")
    for rank, (dist, idx_pos) in enumerate(zip(D[0], I[0]), start=1):
        if idx_pos < 0:
            continue
        m = meta[idx_pos] if idx_pos < len(meta) else {"id": None}
        chunk_id = m.get("id")
        chunk = chunks.get(chunk_id, {})
        text = chunk.get("text", "<no text>")
        source = m.get("source") or chunk.get("source") or "<unknown>"
        page_info = chunk.get("page_index_start") or m.get("page_index_start") or "?"
        print("---")
        print(f"Rank {rank}  idx_pos={idx_pos}  dist={dist:.4f}")
        print(f"Chunk id: {chunk_id}  source: {source}  page: {page_info}")
        excerpt = text.strip().replace("\n", " ")[:800]
        print("Excerpt:", excerpt)
        contains = args.query.lower() in excerpt.lower()
        print("Contains query in excerpt:", contains)


if __name__ == "__main__":
    main()
