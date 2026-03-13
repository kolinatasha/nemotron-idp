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
