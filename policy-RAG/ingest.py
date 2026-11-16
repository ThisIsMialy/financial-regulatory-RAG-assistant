import os
import glob
from pypdf import PdfReader
import chromadb

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 40]


def ingest_pdfs(docs_dir: str = "./docs", db_path: str = "./chroma_db"):
    os.makedirs(docs_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {docs_dir}/. Add 2-3 PDFs and re-run.")
        return

    client = chromadb.PersistentClient(path=db_path)
    # Reset collection on re-ingest to avoid duplicate chunks
    try:
        client.delete_collection("policy_docs")
    except Exception:
        pass
    collection = client.create_collection(
        name="policy_docs",
        metadata={"hnsw:space": "cosine"},  # Cosine distance for clear similarity thresholds
    )

    documents, metadatas, ids = [], [], []
    doc_counter = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        for page_idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            clean_text = " ".join(raw_text.split())
            for chunk in chunk_text(clean_text):
                documents.append(chunk)
                metadatas.append({"source": filename, "page": page_idx})
                ids.append(f"chunk_{doc_counter}")
                doc_counter += 1

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Ingested {doc_counter} chunks from {len(pdf_files)} PDF(s) into {db_path}.")


if __name__ == "__main__":
    ingest_pdfs()