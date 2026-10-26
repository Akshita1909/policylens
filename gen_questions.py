import os
import json
import random
from pypdf import PdfReader
from groq import Groq

PDF_PATH = "doc.pdf"
NUM_QUESTIONS = 15
FIRST_PAGE = 8          # skip title page and the list of amendments
MIN_CHARS = 800         # skip pages with too little text
PASSAGE_CHARS = 700
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

random.seed(7)

PROMPT = """Below is a passage from a regulatory document.
Write ONE specific question that can be answered using only this passage.

Rules:
- The question must make sense on its own. Do not say "this passage" or "the text".
- It should ask about a concrete rule, number, deadline, condition or definition.
- Do not copy long phrases from the passage.
- Return only the question, nothing else.

Passage:
"""


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set in this terminal.")
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    reader = PdfReader(PDF_PATH)

    pages = []
    for num, page in enumerate(reader.pages, start=1):
        if num < FIRST_PAGE:
            continue
        text = " ".join((page.extract_text() or "").split())
        if len(text) >= MIN_CHARS:
            pages.append((num, text))

    chosen = random.sample(pages, min(NUM_QUESTIONS + 5, len(pages)))
    questions = []

    for num, text in chosen:
        if len(questions) >= NUM_QUESTIONS:
            break
        start = random.randint(0, max(0, len(text) - PASSAGE_CHARS))
        passage = text[start:start + PASSAGE_CHARS]
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.3,
                messages=[{"role": "user", "content": PROMPT + passage}],
            )
            q = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"page {num}: LLM error: {e}")
            continue
        if len(q) < 15 or not q.endswith("?"):
            print(f"page {num}: skipped (bad question: {q[:60]!r})")
            continue
        questions.append({"question": q, "page": num})
        print(f"[page {num}] {q}")

    with open("questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nsaved {len(questions)} questions to questions.json")


if __name__ == "__main__":
    main()
