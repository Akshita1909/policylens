# PolicyLens

Ask questions about a pdf and get answers with page numbers. It only answers from the document. If the answer isn't in there, it says so.

Live demo: https://policylens-h6ak32tbim2jl65ddgntjw.streamlit.app/

I built this to understand how RAG systems behave, and mostly to measure how well the retrieval part works instead of just eyeballing a few answers.

## What it does

- Built-in document: the RBI Master Direction on KYC (107 pages)
- Or upload your own text-based pdf (max 15 MB, first 300 pages are read, processed in memory and not saved)
- Every answer cites pages like [p.31] and you can open the exact chunks it used
- If the best match is weak (similarity under 0.30, adjustable in the sidebar), it refuses without calling the LLM

## How it works

    pdf -> text per page -> chunks (800 chars, 100 overlap, page number kept)
        -> embeddings (all-MiniLM-L6-v2) -> Chroma
        -> top 4 chunks for the question
        -> Groq gpt-oss-120b, temperature 0, told to answer only from the context
        -> answer with [p.x] citations

Stack: Python, sentence-transformers, ChromaDB, Groq, Streamlit, pandas.

## Eval

`gen_questions.py` picks random passages from the pdf and has the LLM write one question per passage, so the correct page is known. It generated 15, I read them and removed 3 that were vague, which left 12. `run_eval.py` then checks whether the right page shows up in the top k results.

| chunking | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| 800 / 100 | 381 | 0.75 | 0.92 | 1.00 | 0.85 |
| 400 / 50 | 697 | 0.67 | 0.92 | 1.00 | 0.80 |

What this does and doesn't show:

- 12 questions is small. The gap between the two settings is one question, so I wouldn't say either is better.
- The questions were written from the passages themselves, so they share wording with the answer text. Real users paraphrase, so real numbers would be lower.
- Only retrieval is measured here. I haven't scored whether the final answers are correct.

Next steps: a bigger set with paraphrased questions, and a groundedness check on the answers.

## Known limits

- Scanned pdfs (images) don't work, there is no text to read
- The 0.30 refusal threshold came from looking at a few examples (off-topic questions scored about 0.13 to 0.27, relevant ones 0.5 and above). It is not tuned, so a vague question about the document can get refused
- No reranker, no hybrid search
- Tables in pdfs get flattened by text extraction

## Run it locally

    pip install -r requirements.txt
    export GROQ_API_KEY="your key"
    streamlit run app.py

To build the index for another pdf in the built-in mode:

    python ingest.py yourfile.pdf 800 100

## Files

- `app.py` the Streamlit app
- `ingest.py` reads a pdf, chunks it, embeds it, stores it in Chroma
- `gen_questions.py` makes the eval questions
- `run_eval.py` scores retrieval for each chunking setting
- `questions.json`, `eval_results.csv` eval inputs and results
