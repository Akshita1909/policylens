import os
import chromadb
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
NOT_FOUND = "Not found in the document."

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
    st.error("GROQ_API_KEY is not set in this terminal. Stop the app, export it, and run again.")
    st.stop()

client_db = load_db()
collections = sorted(c.name for c in client_db.list_collections())
if not collections:
    st.error("No collections found. Run: python ingest.py doc.pdf 800 100")
    st.stop()

default_index = collections.index("doc_800_100") if "doc_800_100" in collections else 0

with st.sidebar:
    st.header("Settings")
    collection_name = st.selectbox("Chunk setting (collection)", collections, index=default_index)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    min_sim = st.slider("Refuse below this similarity", 0.0, 0.8, 0.30, 0.05)
    st.caption(f"LLM: {LLM_MODEL}")
    st.caption(f"Embeddings: {EMBED_MODEL}")

embedder = load_embedder()
collection = client_db.get_collection(collection_name)

tab_ask, tab_eval = st.tabs(["Ask", "Eval results"])

with tab_ask:
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
        "Questions were auto-generated from random passages, so the correct page is known. "
        "hit@k = the right page appears in the top k chunks. MRR = how high it ranks on average."
    )
    if os.path.exists("eval_results.csv"):
        st.dataframe(pd.read_csv("eval_results.csv"), hide_index=True)
        st.caption("Small question set: differences between configs are within noise.")
    else:
        st.info("Run python run_eval.py to generate eval_results.csv")
