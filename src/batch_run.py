import os
from rag import (
    load_chunks,
    build_or_load_embeddings,
    build_index,
    retrieve,
    answer_from_context,
)
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def read_questions(path: str) -> list[str]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                qs.append(q)
    return qs


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    q_path = os.path.join(base_dir, "questions.txt")

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"questions.txt not found at: {q_path}")

    print("Loading docs and building index...")
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    emb = build_or_load_embeddings(chunks, model)
    index = build_index(emb)

    questions = read_questions(q_path)
    print(f"\nRunning {len(questions)} questions...\n")

    for i, q in enumerate(questions, start=1):
        retrieved = retrieve(model, index, chunks, q, k=8)
        out = answer_from_context(q, retrieved, chunks)

        print("=" * 100)
        print(f"Q{i}: {q}\n")
        print(out)
        print("=" * 100)
        print()


if __name__ == "__main__":
    main()
