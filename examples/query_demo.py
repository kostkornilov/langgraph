"""Example: query the FAISS index locally and print results with links."""
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
FAISS_DIR = ROOT / "data" / "faiss"

def main():
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        print("LangChain FAISS wrapper not available. This example expects langchain with FAISS support.")
        return

    store = FAISS.load_local(str(FAISS_DIR))
    query = input("Enter query: ")
    docs = store.similarity_search(query, k=5)
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source")
        page_start = meta.get("page_index_start")
        link = f"/pdf/{source}#page={int(page_start)+1}" if source and page_start is not None else None
        print("---")
        print("Source:", source, "pages:", meta.get("page_index_start"), "-", meta.get("page_index_end"))
        print("Link:", link)
        print("Excerpt:", d.page_content[:400])


if __name__ == "__main__":
    main()
