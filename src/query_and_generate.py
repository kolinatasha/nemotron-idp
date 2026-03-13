"""Query + Generate pipeline using retrieval + NVIDIA NIM generation.

- Retrieves documents from Milvus (or another vector store)
- Optionally reranks using visual reranker
- Calls NVIDIA NIM generation endpoint to produce citation-backed answers

NOTE: Replace `NIM_ENDPOINT` and payload with the exact contract required by your NIM deployment.
"""
from typing import List, Dict, Any
import os
import requests

NIM_ENDPOINT = "https://api.nvidia.com/v1/nim/generate"  # placeholder


def generate_with_nim(prompt: str, contexts: List[Dict[str, Any]], api_key: str, max_tokens: int = 512) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Build a structured prompt with citations
    context_text = "\n\n".join([f"Source: {c.get('source', 'unknown')}\n{c.get('text','')}" for c in contexts])
    body = {
        "model": "nvidia/llama-3.3-nemotron-super-49b",
        "input": {
            "prompt": prompt,
            "context": context_text,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "return_citations": True
        }
    }
    resp = requests.post(NIM_ENDPOINT, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("Set NVIDIA_API_KEY to call the NIM generation API")
    else:
        out = generate_with_nim("Summarize revenue trends.", [{"source":"doc1.pdf","text":"Revenue rose 20% in 2025 compared to 2024."}], api_key)
        print(out)
