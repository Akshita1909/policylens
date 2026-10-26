import json
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTIONS = ["doc_800_100", "doc_400_50"]
MAX_K = 5


def main():
    with open("questions.json") as f:
        questions = json.load(f)

    print(f"{len(questions)} questions")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    q_embs = model.encode([q["question"] for q in questions]).tolist()

    summary = []
    misses = []

    for name in COLLECTIONS:
        collection = client.get_collection(name)
        hit1 = hit3 = hit5 = 0
        rr_sum = 0.0

        for q, emb in zip(questions, q_embs):
            res = collection.query(query_embeddings=[emb], n_results=MAX_K)
            pages = [m["page"] for m in res["metadatas"][0]]
            gold = q["page"]

            if gold in pages:
                rank = pages.index(gold) + 1
                rr_sum += 1.0 / rank
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
            else:
                misses.append({
                    "collection": name,
                    "question": q["question"],
                    "gold_page": gold,
                    "retrieved_pages": pages,
                })

        n = len(questions)
        summary.append({
            "collection": name,
            "chunks": collection.count(),
            "hit@1": round(hit1 / n, 2),
            "hit@3": round(hit3 / n, 2),
            "hit@5": round(hit5 / n, 2),
            "MRR": round(rr_sum / n, 2),
        })

    df = pd.DataFrame(summary)
    print("\n=== retrieval results ===")
    print(df.to_string(index=False))
    df.to_csv("eval_results.csv", index=False)

    print("\n=== misses (right page not in top 5) ===")
    if not misses:
        print("none")
    for m in misses:
        print(f"[{m['collection']}] gold p.{m['gold_page']} got {m['retrieved_pages']}")
        print(f"   {m['question']}")


if __name__ == "__main__":
    main()
