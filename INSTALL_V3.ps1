$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path 'app.py')) { throw 'Extract the update into the existing project folder.' }
python -c "import torch, streamlit, fitz, faiss, sentence_transformers, rouge_score; print(torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Activate your existing project virtual environment first.' }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
python -m streamlit run app.py
