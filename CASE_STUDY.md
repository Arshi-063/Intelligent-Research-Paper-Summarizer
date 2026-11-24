# Case Study: Automated Literature Review of Diabetes Risk Prediction Research

## Problem
A graduate researcher must read five or more long research papers and extract comparable objectives, datasets, methods, findings, limitations, and future work. Manual review is slow, and general-purpose generative systems may hallucinate results or confuse papers.

## Stakeholders
Postgraduate students, academic researchers, supervisors, and research librarians.

## Inputs and outputs
Input: open-access text-based academic PDFs. Output: six evidence-grounded categories per paper, source page references, and downloadable comparison CSV. PDFs with only scanned images need OCR.

## Functional requirements
Multi-PDF upload; extraction; page-aware chunking; semantic retrieval; structured summary; evidence inspection; CSV export; graceful handling of malformed or scanned PDFs.

## Nonfunctional requirements
Faithfulness, traceability, reproducibility, privacy, measured latency, and acceptable memory usage.

## Methodology
1. Collect licensed PDFs and annotate reference facts.
2. Extract PDF text with PyMuPDF and retain page numbers.
3. Chunk text with overlap; embed with MiniLM and index with FAISS.
4. Retrieve top-k passages for six research questions.
5. Generate using a local transformer with explicit evidence-only prompting, or display evidence excerpts in fallback mode.
6. Validate claims against original pages.
7. Upgrade to a verified long-context model; integrate NeMo and TensorRT-LLM on compatible hardware; benchmark fairly.

## Hypotheses
H1: Retrieval grounding improves factual support compared with ungrounded generation.
H2: Section-aware long-context processing reduces omissions from distant sections.
H3: On supported hardware and model, TensorRT-LLM can improve inference performance; magnitude must be measured.

## Experimental comparisons
A: extractive evidence-only baseline. B: local LLM without retrieval. C: local LLM + RAG. D: long-context + RAG. E: same supported model on baseline GPU backend vs TensorRT-LLM. Hold evaluation documents constant, use consistent prompts, and disclose hardware and inference settings.

## Metrics
Retrieval precision@k, recall@k; human-checked grounded-claim percentage; citation correctness; ROUGE-L and BERTScore against human summaries; median and p95 latency, tokens/sec, GPU memory.

## Limitations
The supplied runnable prototype is not a completed NeMo or TensorRT-LLM deployment. It is a baseline foundation. Citations are page-level retrieval pointers, not guaranteed claim-level citations. PDFs with complex columns, tables, formulas, or scans may parse poorly. No benchmark numbers are claimed before testing.

## Expected outcomes
A demonstrable dashboard that retrieves supporting passages and produces structured outputs. Subsequent experiments will establish whether long-context and NVIDIA optimization deliver measurable benefits.
