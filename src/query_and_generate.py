"""Query + Generate pipeline using retrieval + NVIDIA NIM generation.

- Retrieves documents from Milvus (or another vector store)
- Optionally reranks using visual reranker
- Calls NVIDIA NIM chat-completions endpoint to produce citation-backed answers
"""
from typing import List, Dict, Any, Optional
import os
import requests

DEFAULT_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NIM_MODEL = "meta/llama-3.1-8b-instruct"

NIM_ENDPOINT = os.environ.get("NIM_ENDPOINT", DEFAULT_NIM_ENDPOINT)
NIM_MODEL = os.environ.get("NIM_MODEL", DEFAULT_NIM_MODEL)


def build_context_text(contexts: List[Dict[str, Any]]) -> str:
    return "\n\n".join([f"Source: {c.get('source', 'unknown')}\n{c.get('text', '')}" for c in contexts])


def generate_with_nim(
    prompt: str,
    contexts: List[Dict[str, Any]],
    api_key: str,
    max_tokens: int = 512,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 120,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    context_text = build_context_text(contexts)

    # Keep prompt structure deterministic to simplify testing.
    user_prompt = (
        "Use ONLY the supplied context. If the answer is missing, say so clearly. "
        "Include source citations in your answer.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Context:\n{context_text}"
    )

    body = {
        "model": model or NIM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise enterprise assistant that cites provided sources.",
            },
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    target_endpoint = endpoint or NIM_ENDPOINT
    resp = requests.post(target_endpoint, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("Set NVIDIA_API_KEY to call the NIM generation API")
    else:
        out = generate_with_nim(
            "Summarize revenue trends.",
            [{"source": "doc1.pdf", "text": "Revenue rose 20% in 2025 compared to 2024."}],
            api_key,
        )
        print(out)
