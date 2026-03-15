import os
import re
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8


@dataclass
class Chunk:
    doc_name: str
    page_num: int
    chunk_id: int
    text: str


def read_pdf_pages(path: str) -> List[str]:
    reader = PdfReader(path)
    return [(p.extract_text() or "") for p in reader.pages]


def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    text = clean_text(text)
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def load_chunks() -> List[Chunk]:
    pdfs = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {os.path.abspath(DOCS_DIR)}")

    all_chunks: List[Chunk] = []
    for pdf_path in pdfs:
        doc_name = os.path.basename(pdf_path)
        pages = read_pdf_pages(pdf_path)

        for page_i, page_text in enumerate(pages, start=1):
            parts = chunk_text(page_text)
            for ci, part in enumerate(parts):
                all_chunks.append(
                    Chunk(doc_name=doc_name, page_num=page_i, chunk_id=ci, text=part)
                )

    return all_chunks


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return vecs.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def keyword_boost(
    chunks: List[Chunk], query: str, limit: int = 30
) -> List[Tuple[int, float]]:
    q = query.lower()

    if "email" in q:
        keywords = ["email", "e-mail", "@", "contact", "office", "phone"]
    elif any(w in q for w in ["late", "submission", "deadline", "penalty"]):
        keywords = ["late", "accepted", "submission", "deadline", "penalty"]
    elif any(w in q for w in ["grading", "grade", "weights", "weighted"]):
        keywords = [
            "grading",
            "weighted",
            "midterm",
            "final",
            "project",
            "homework",
            "%",
        ]
    elif any(w in q for w in ["prereq", "prerequisite"]):
        keywords = ["prerequisite", "prerequisites", "recommended", "consent"]
    elif any(w in q for w in ["textbook", "textbooks", "required", "isbn"]):
        keywords = ["textbooks", "required", "optional", "isbn", "press"]
    elif any(w in q for w in ["office hours"]):
        keywords = ["office hours", "mw", "tr"]
    elif any(w in q for w in ["workload", "hours per week", "hours/week"]):
        keywords = ["hours/week", "hours per week", "workload"]
    elif any(w in q for w in ["meeting time", "class time", "schedule", "meets"]):
        keywords = ["meeting", "meets", "time", "schedule"]
    else:
        keywords = re.findall(r"[a-zA-Z]{3,}", q)[:6]

    scored: List[Tuple[int, float]] = []
    for i, ch in enumerate(chunks):
        t = ch.text.lower()
        score = 0.0
        for kw in keywords:
            if kw in t:
                score += 1.0
        if "@" in t:
            score += 0.5
        if score > 0:
            scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def retrieve(
    model, index, chunks: List[Chunk], query: str, k: int = TOP_K
) -> List[Tuple[Chunk, float]]:
    qv = embed_texts(model, [query])
    faiss.normalize_L2(qv)
    sem_scores, sem_ids = index.search(qv, max(k, 10))

    combined = {}

    for idx, score in zip(sem_ids[0], sem_scores[0]):
        if idx == -1:
            continue
        combined[int(idx)] = max(combined.get(int(idx), 0.0), float(score))

    for idx, kw_score in keyword_boost(chunks, query, limit=30):
        combined[idx] = combined.get(idx, 0.0) + (0.08 * kw_score)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(chunks[i], float(s)) for i, s in ranked]


def find_header_value(
    all_chunks: List[Chunk], header: str
) -> Tuple[Optional[str], Optional[Chunk]]:
    # tolerate spacing quirks from PDF extraction: "Pre requisites"
    header_regex = r"\b" + r"\s*".join(map(re.escape, header)) + r"\s*:\s*(.+)"
    for ch in all_chunks:
        m = re.search(header_regex, ch.text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), ch
    return None, None


def extract_email(all_chunks: List[Chunk]) -> Tuple[Optional[str], Optional[Chunk]]:
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    for ch in all_chunks:
        m = re.search(email_pattern, ch.text)
        if m:
            return m.group(0), ch
    return None, None


def extract_hours_per_week(
    all_chunks: List[Chunk],
) -> Tuple[Optional[str], Optional[Chunk]]:
    # examples: "6+ hours/week", "6 hours/week", "6+ hours / week"
    pat = r"(\d+\s*\+\s*hours\s*/\s*week|\d+\s*hours\s*/\s*week|\d+\s*\+\s*hours\s*per\s*week|\d+\s*hours\s*per\s*week)"
    for ch in all_chunks:
        m = re.search(pat, ch.text, flags=re.IGNORECASE)
        if m:
            return m.group(1), ch
    return None, None


def extract_meeting_time(
    all_chunks: List[Chunk],
) -> Tuple[Optional[str], Optional[Chunk]]:
    # Try to find a day+time pattern near early syllabus pages
    # Examples: "MW 16:30 - 17:30" or "Monday, May 18, 2015 17:15-19:30"
    day_time_pat = r"\b(MW|TR|M|T|W|R|F|Monday|Tuesday|Wednesday|Thursday|Friday)\b.*?(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})"
    for ch in all_chunks:
        if "syllabus" in ch.doc_name.lower() and ch.page_num <= 2:
            m = re.search(day_time_pat, ch.text)
            if m:
                return m.group(0), ch
    return None, None


def answer_from_context(
    query: str, retrieved: List[Tuple[Chunk, float]], all_chunks: List[Chunk]
) -> str:
    q = query.lower()

    # ===== Structured extraction for common syllabus fields =====
    if "email" in q:
        email, ch = extract_email(all_chunks)
        if email and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Instructor email: {email}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )
        return "Email not found in the provided documents."

    if "phone" in q or "telephone" in q:
        val, ch = find_header_value(all_chunks, "PHONE")
        if val and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Instructor phone: {val}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if ("office" in q) and ("hours" not in q):
        val, ch = find_header_value(all_chunks, "OFFICE")
        if val and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Instructor office: {val}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if "office hour" in q:
        val, ch = find_header_value(all_chunks, "Office Hours")
        if val and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Office hours: {val}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if "prereq" in q or "prerequisite" in q or "taken before" in q:
        val, ch = find_header_value(all_chunks, "Prerequisites")
        if val and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Prerequisites: {val}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if "textbook" in q or "textbooks" in q or "required" in q:
        val, ch = find_header_value(all_chunks, "Textbooks")
        if val and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Textbooks: {val}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if "workload" in q or "hours per week" in q or "hours/week" in q:
        hrs, ch = extract_hours_per_week(all_chunks)
        if hrs and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Expected workload: {hrs}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    if "meeting time" in q or "class time" in q or ("when" in q and "meet" in q):
        mt, ch = extract_meeting_time(all_chunks)
        if mt and ch:
            return "\n".join(
                [
                    f"Question: {query}\n",
                    "Grounded answer (from documents):",
                    f"- Meeting time (found): {mt}\n",
                    "Citation:",
                    f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                ]
            )

    # Late policy sentence extraction (works well)
    if any(w in q for w in ["late", "submission", "deadline", "penalty"]):
        for ch, _score in retrieved:
            sents = re.split(r"(?<=[.!?])\s+", ch.text)
            for s in sents:
                if "late" in s.lower():
                    return "\n".join(
                        [
                            f"Question: {query}\n",
                            "Grounded answer (from documents):",
                            f"- {s.strip()}\n",
                            "Citation:",
                            f"[1] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id}",
                        ]
                    )

    # ===== Evidence-first fallback (safe / no hallucination) =====
    if not retrieved:
        return "Not found in the provided documents."

    top_score = retrieved[0][1]
    if top_score < 0.20:
        return "Not found in the provided documents (low match). Try rephrasing with keywords from the documents."

    lines = []
    lines.append(f"Question: {query}\n")
    lines.append("Evidence (top matches):")

    for rank, (ch, score) in enumerate(retrieved[:3], start=1):
        snippet = ch.text[:350].strip()
        lines.append(
            f"[{rank}] {ch.doc_name} (page {ch.page_num})#chunk{ch.chunk_id} (score={score:.3f})"
        )
        lines.append(f"    {snippet}...\n")

    lines.append("Answer guidance:")
    lines.append(
        "- The answer should be supported by the evidence above. (Transparency over guessing.)"
    )
    return "\n".join(lines)


def build_or_load_embeddings(
    chunks: List[Chunk], model: SentenceTransformer
) -> np.ndarray:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    emb_path = os.path.join(cache_dir, "embeddings.npy")
    meta_path = os.path.join(cache_dir, "embeddings_meta.txt")

    # A simple fingerprint: doc names + number of chunks
    fingerprint = f"{len(chunks)}|" + "|".join(sorted({c.doc_name for c in chunks}))

    if os.path.exists(emb_path) and os.path.exists(meta_path):
        try:
            old_fp = open(meta_path, "r", encoding="utf-8").read().strip()
            if old_fp == fingerprint:
                emb = np.load(emb_path).astype("float32")
                return emb
        except Exception:
            pass  # fall through to rebuild

    emb = embed_texts(model, [c.text for c in chunks])
    np.save(emb_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(fingerprint)

    return emb


def main():
    print("Loading PDFs from:", os.path.abspath(DOCS_DIR))
    chunks = load_chunks()
    print(
        f"Loaded {len(chunks)} chunks from {len(set(c.doc_name for c in chunks))} PDFs"
    )

    model = SentenceTransformer(MODEL_NAME)
    emb = build_or_load_embeddings(chunks, model)
    index = build_index(emb)

    print("\nRAG is ready. Type a question (or 'exit').\n")
    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(model, index, chunks, q, k=TOP_K)
        print()
        print(answer_from_context(q, retrieved, chunks))
        print("-" * 80)


if __name__ == "__main__":
    main()
