# RAG Doc Q&A Demo (PDFs + Citations)

This is a small-scale Retrieval-Augmented QA demo over local PDF documents.
It supports:
- PDF ingestion + chunking
- Embedding-based retrieval (FAISS)
- Hybrid retrieval boost (semantic + keyword)
- Grounded answers for common syllabus-style fields (email, phone, office, office hours, prerequisites, textbooks, late policy)
- Evidence-first fallback with citations (avoids hallucination)
- For conceptual questions, returns top evidence passages + citations (no free-form generation)
- Embedding caching for faster reruns

## Folder structure
- docs/        → PDF inputs (syllabus.pdf, notes.pdf, reading.pdf)
- src/rag.py   → interactive CLI
- src/batch_run.py → batch runner for questions.txt
- questions.txt → one question per line
- cache/       → saved embeddings for faster runs

## Setup (Windows)
```powershell
cd C:\rag-docqa-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pypdf sentence-transformers faiss-cpu numpy
