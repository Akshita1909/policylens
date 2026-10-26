import sys
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"


def chunk_text(text, size, overlap):
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if len(piece) > 50:
            chunks.append(piece)
    return chunks


def main():
    if len(sys.argv) != 4:
        print("usage: python ingest.py <file.pdf> <chunk_size> <overlap>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    size = int(sys.argv[2])
    overlap = int(sys.argv[3])
    collection_name = f"doc_{size}_{overlap}"

    print(f"reading {pdf_path} ...")
    reader = PdfReader(pdf_path)

    docs, metas = [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        for chunk in chunk_text(text, size, overlap):
            docs.append(chunk)
            metas.append({"page": page_num, "source": pdf_path})

    print(f"{len(reader.pages)} pages -> {len(docs)} chunks (size={size}, overlap={overlap})")
    if not docs:
        print("no text found. this pdf may be scanned images, try a different pdf.")
        sys.exit(1)

    print("loading embedding model (first run downloads it, takes a minute) ...")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch = 64
    for i in range(0, len(docs), batch):
        part_docs = docs[i:i + batch]
        part_metas = metas[i:i + batch]
        part_ids = [f"chunk_{j}" for j in range(i, i + len(part_docs))]
        embeddings = model.encode(part_docs, show_progress_bar=False).tolist()
        collection.add(
            ids=part_ids,
            documents=part_docs,
            embeddings=embeddings,
            metadatas=part_metas,
        )
        print(f"stored {min(i + batch, len(docs))}/{len(docs)}")

    print(f"done. collection: {collection_name}")


if __name__ == "__main__":
    main()
