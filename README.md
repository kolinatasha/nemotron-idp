# Nemotron RAG - Intelligent Document Processing (IDP)

Production-oriented scaffold for converting complex enterprise documents (text, tables, charts) into a multimodal RAG knowledge base using NVIDIA Nemotron models.

## Included
- Layout-aware ingestion (`src/ingest.py`)
- Multimodal embedding + Milvus indexing (`src/embed_and_store.py`)
- Visual-aware reranking (`src/rerank.py`)
- Retrieval + NIM generation (`src/query_and_generate.py`)
- Demo notebook (`notebooks/intelligent_document_processing_pipeline.ipynb`)
- Smoke tests (`tests/test_smoke.py`)
- CI workflow (`.github/workflows/ci.yml`)

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

## Notes
- Set `NVIDIA_API_KEY` for NIM generation.
- Configure Milvus URI in scripts before running indexing/search.
- Optional overrides:
	- `NIM_ENDPOINT` (defaults to `https://integrate.api.nvidia.com/v1/chat/completions`)
	- `NIM_MODEL` (defaults to `meta/llama-3.1-8b-instruct`)

## GitHub Actions Secret
For GitHub Actions, store the secret as `NVIDIA_NIM_API_KEY` and let workflows map it to `NVIDIA_API_KEY` at runtime.

1. Open your repository on GitHub.
2. Go to Settings > Secrets and variables > Actions.
3. Add a repository or environment secret named `NVIDIA_NIM_API_KEY`.
4. Add optional environment variables `NIM_ENDPOINT` and `NIM_MODEL` if you need non-default runtime values.

The workflows map this secret to the `NVIDIA_API_KEY` environment variable expected by `src/query_and_generate.py`.

## Staging And Production Environments
Use GitHub Environments to separate secrets and approvals between staging and production.

1. Go to Settings > Environments and create `staging` and `production`.
2. In each environment, add secret `NVIDIA_NIM_API_KEY`.
3. In each environment, add variables `NIM_ENDPOINT` and `NIM_MODEL` if needed.
4. Configure required reviewers for `production` before deployment or integration runs.
5. Trigger `.github/workflows/nim-integration.yml` manually and select the target environment.

`ci.yml` stays fast and runs smoke tests only. `nim-integration.yml` runs a live API check against the selected environment with its own secret and approval controls.
