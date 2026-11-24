# STEP-BY-STEP WINDOWS GUIDE

## First run
1. Install Python **3.11 (64-bit)** from python.org and enable **Add Python to PATH**.
2. Extract this ZIP into `D:\Research_Paper_Summarizer` (or another short path).
3. Double-click `RUN_WINDOWS.bat`. The first installation and model download require internet and can take time.
4. Wait for the message `Local URL: http://localhost:8501` and open it in Chrome.
5. Upload 2–5 **text-selectable** academic PDFs (not scanned images).
6. Choose **Evidence only** first. Click **Analyze papers** and inspect each source passage.
7. Download the CSV and TXT files.
8. Next time, double-click `START_APP.bat`.

## Optional generated summaries
Choose **Local Hugging Face LLM** and use `Qwen/Qwen2.5-1.5B-Instruct`. The first run downloads the model; it can be slow or run out of memory on an 8 GB laptop. If so, stay with evidence-only mode or use a GPU computer. This mode uses a truncated prompt (4096 tokens), not a complete long-context architecture.

## Alternative command line
Open Command Prompt in the extracted project directory:
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

## NVIDIA extension (advanced; not automatically implemented)
NeMo requires a compatible NVIDIA GPU/Linux environment and model-specific configuration. TensorRT-LLM needs supported CUDA/GPU drivers, a supported model and a separately deployed inference backend. After deployment, select **Compatible API endpoint**, enter its URL and served model ID. The endpoint adapter is generic and **does not itself install NeMo or TensorRT-LLM**. Use the official NVIDIA compatibility matrices and instructions for the exact GPU and versions.

## Troubleshooting
- `No suitable Python runtime`: install Python 3.11, then run `py -0p`.
- `py not recognized`: reinstall Python with launcher and PATH enabled.
- `No module named streamlit`: activate `.venv` and run `pip install -r requirements.txt`.
- PDF shows no text: use a text-selectable PDF; scanned documents need OCR.
- Model download error: check internet and retry.
- Out-of-memory: use Evidence only mode or a suitable GPU environment.
- `faiss` import error: use a clean 64-bit Python 3.11 environment; check pip output.
- Port occupied: `streamlit run app.py --server.port 8502`.
