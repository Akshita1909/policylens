import io
import os
import hashlib
import chromadb
import pandas as pd
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
NOT_FOUND = "Not found in the document."

BUILTIN = "Built-in: RBI KYC Master Direction"
UPLOAD = "Upload my own PDF"

MAX_UPLOAD_MB = 15
MAX_PAGES = 300
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SYSTEM_PROMPT = """You answer questions using ONLY the context excerpts provided.
Rules:
- If the answer is not in the context, reply exactly: "Not found in the document."
- Do not use outside knowledge and do not guess.
- Cite the page number after each claim using plain square brackets like [p.12].
- Keep the answer short and specific."""

st.set_page_config(page_title="PolicyLens", page_icon="🔎", layout="wide")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_db():
    return chromadb.PersistentClient(path=DB_PATH)


@st.cache_resource
def load_mem_db():
    return chromadb.EphemeralClient()


def chunk_text(text, size, overlap):
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if len(piece) > 50:
            chunks.append(piece)
    return chunks


def build_index(data, filename, file_hash):
    reader = PdfReader(io.BytesIO(data))
    total_pages = len(reader.pages)
    pages_read = min(total_pages, MAX_PAGES)

    docs, metas = [], []
    for page_num in range(1, pages_read + 1):
        text = " ".join((reader.pages[page_num - 1].extract_text() or "").split())
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            docs.append(chunk)
            metas.append({"page": page_num, "source": filename})

    if not docs:
        return None

    mem = load_mem_db()
    name = f"upload_{file_hash[:16]}"
    try:
        mem.delete_collection(name)
    except Exception:
        pass
    col = mem.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    embedder = load_embedder()
    bar = st.progress(0.0, text="Embedding chunks...")
    batch = 64
    for i in range(0, len(docs), batch):
        part_docs = docs[i:i + batch]
        part_metas = metas[i:i + batch]
        part_ids = [f"chunk_{j}" for j in range(i, i + len(part_docs))]
        embs = embedder.encode(part_docs, show_progress_bar=False).tolist()
        col.add(ids=part_ids, documents=part_docs, embeddings=embs, metadatas=part_metas)
        done = min(i + batch, len(docs))
        bar.progress(done / len(docs), text=f"Embedding chunks... {done}/{len(docs)}")
    bar.empty()

    return {
        "name": name,
        "filename": filename,
        "pages": total_pages,
        "pages_read": pages_read,
        "chunks": len(docs),
    }


def retrieve(collection, embedder, question, top_k):
    q_emb = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({"text": doc, "page": meta["page"], "score": 1 - dist})
    return chunks


def generate_answer(question, chunks):
    context = "\n\n".join(f"[p.{c['page']}] {c['text']}" for c in chunks)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = resp.choices[0].message.content or ""
    return text.replace("【", "[").replace("】", "]").strip()


st.title("🔎 PolicyLens")
st.caption("Ask questions about a document. Every answer shows its sources.")

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to the app secrets (cloud) or export it in the terminal (local).")
    st.stop()

client_db = load_db()
collections = sorted(c.name for c in client_db.list_collections())
if not collections:
    st.error("No built-in collections found. Run: python ingest.py doc.pdf 800 100")
    st.stop()

default_index = collections.index("doc_800_100") if "doc_800_100" in collections else 0

with st.sidebar:
    st.header("Settings")
    source = st.radio("Document", [BUILTIN, UPLOAD])
    collection_name = collections[default_index]
    if source == BUILTIN:
        collection_name = st.selectbox("Chunk setting (collection)", collections, index=default_index)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    min_sim = st.slider("Refuse below this similarity", 0.0, 0.8, 0.30, 0.05)
    st.caption(f"LLM: {LLM_MODEL}")
    st.caption(f"Embeddings: {EMBED_MODEL}")

embedder = load_embedder()

if "indexes" not in st.session_state:
    st.session_state["indexes"] = {}

tab_ask, tab_eval = st.tabs(["Ask", "Eval results"])

with tab_ask:
    collection = None

    if source == BUILTIN:
        collection = client_db.get_collection(collection_name)
        st.caption("Answering from: RBI Master Direction on KYC (built-in)")
    else:
        st.caption(
            f"Text-based PDFs only. Max {MAX_UPLOAD_MB} MB, first {MAX_PAGES} pages are read. "
            "Your file is processed in memory and not saved. Please don't upload sensitive documents."
        )
        up = st.file_uploader("Upload a PDF", type=["pdf"])

        if up is None:
            st.info("Upload a PDF to start asking questions about it.")
        elif up.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {up.size / 1024 / 1024:.1f} MB. The limit is {MAX_UPLOAD_MB} MB.")
        else:
            data = up.getvalue()
            file_hash = hashlib.sha256(data).hexdigest()
            info = st.session_state["indexes"].get(file_hash)

            if info is not None:
                try:
                    collection = load_mem_db().get_collection(info["name"])
                except Exception:
                    st.session_state["indexes"].pop(file_hash, None)
                    info = None

            if info is None:
                try:
                    with st.spinner("Reading and indexing your PDF..."):
                        info = build_index(data, up.name, file_hash)
                except Exception as e:
                    info = None
                    st.error(f"Could not read this PDF: {e}")
                else:
                    if info is None:
                        st.error("No readable text found. This PDF may be a scan (images). Try a text-based PDF.")
                    else:
                        st.session_state["indexes"][file_hash] = info
                        collection = load_mem_db().get_collection(info["name"])

            if info is not None and collection is not None:
                msg = f"Indexed {info['filename']}: {info['pages_read']} pages, {info['chunks']} chunks."
                if info["pages"] > info["pages_read"]:
                    msg += f" (PDF has {info['pages']} pages, only the first {info['pages_read']} were read.)"
                st.success(msg)

    if collection is not None:
        question = st.text_input("Your question")

        if question:
            with st.spinner("Searching and answering..."):
                chunks = retrieve(collection, embedder, question, top_k)
                best = chunks[0]["score"] if chunks else 0.0

                if best < min_sim:
                    answer = NOT_FOUND
                    st.info(f"Best similarity {best:.2f} is below the {min_sim:.2f} threshold, so the LLM was not called.")
                else:
                    try:
                        answer = generate_answer(question, chunks)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
                        st.stop()

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for i, c in enumerate(chunks, start=1):
                with st.expander(f"Source {i}: page {c['page']} (similarity {c['score']:.2f})"):
                    st.write(c["text"])

with tab_eval:
    st.subheader("Retrieval eval")
    st.caption(
        "Measured on the built-in RBI KYC document. Questions were auto-generated from random passages, "
        "so the correct page is known. hit@k = the right page appears in the top k chunks. "
        "MRR = how high it ranks on average."
    )
    if os.path.exists("eval_results.csv"):
        st.dataframe(pd.read_csv("eval_results.csv"), hide_index=True)
        st.caption("Small question set (12): differences between configs are within noise.")
    else:
        st.info("Run python run_eval.py to generate eval_results.csv")
