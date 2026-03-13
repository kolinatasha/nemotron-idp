import os

import pytest
import requests

from src.query_and_generate import generate_with_nim


def test_nim_generation_live_call():
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY is not configured")

    contexts = [
        {
            "source": "finance_q2.txt",
            "text": "Revenue increased 20% year-over-year while operating margin improved.",
        }
    ]

    try:
        response = generate_with_nim(
            prompt="Did revenue increase? Answer in one sentence with source.",
            contexts=contexts,
            api_key=api_key,
            max_tokens=80,
            temperature=0.0,
            timeout=60,
        )
    except requests.RequestException as exc:
        pytest.fail(f"NIM integration request failed: {exc}")

    assert isinstance(response, dict)
    assert response, "Expected non-empty JSON response from NIM"
