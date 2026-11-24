# Intelligent Research Paper Summarization — End-to-End Case Study App

A runnable Streamlit research-paper comparison app with page-aware PDF parsing, sentence-transformer embeddings, FAISS retrieval, evidence inspection, optional Hugging Face text generation, optional compatible inference-server connector, CSV/TXT exports, and runtime measurements.

**Start here:** `STEP_BY_STEP_WINDOWS.md` or `RUN_WINDOWS.bat`.

**Case study:** 2–5 openly accessible diabetes prediction research papers. Do not invent paper findings; use actual uploaded PDFs. The included sample folder is intentionally empty because distributing third-party copyrighted papers requires appropriate rights.

**Scope and limitations:** Core RAG and interface are implemented. NeMo training/customization, a genuinely long-context model, and TensorRT-LLM deployment are advanced extension tasks, not implemented or benchmarked in this ZIP. The default evidence mode extracts snippets; it does not generate an abstractive summary. The local LLM mode has a 4096-token prompt cap. Page references are retrieval provenance, not guaranteed claim-level citation verification. OCR is not included.

See `CASE_STUDY.md` for objectives and evaluation plan. Run `python -m pytest -q` for included tests.
