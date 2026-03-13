"""Visual-aware reranker (cross-encoder style)

Given a query and candidate docs (text + optional chart images), produce a ranked list.
Replace with `nvidia/llama-nemotron-rerank-vl-1b-v2` usage.
"""
from typing import List, Dict, Any
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# Fallback to sentence-transformers CrossEncoder for CPU-friendly reranking
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class Reranker:
    def __init__(self, model_name: str = "nvidia/llama-nemotron-rerank-vl-1b-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if (AutoModelForSequenceClassification is not None and hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available()) else "cpu")
        self.tokenizer = None
        self.model = None
        self.cross_encoder = None

        # Try HF cross-encoder
        try:
            if AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                if self.device.startswith("cuda") and __import__('torch').cuda.is_available():
                    self.model.to('cuda')
        except Exception:
            self.tokenizer = None
            self.model = None

        # Fallback to sentence-transformers CrossEncoder
        if self.model is None and CrossEncoder is not None:
            try:
                # use a small public cross-encoder for CPU fallback
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self.cross_encoder = None

    def score(self, query: str, candidates: List[Dict[str, Any]]) -> List[float]:
        texts = []
        for c in candidates:
            txt = c.get('text') or c.get('snippet') or ''
            texts.append([query, txt])

        # Use CrossEncoder when available
        if self.cross_encoder is not None:
            scores = self.cross_encoder.predict(texts)
            return [float(s) for s in scores]

        # If HF tokenizer+model is available, run a simple classification forward pass
        if self.tokenizer is not None and self.model is not None:
            import torch
            self.model.eval()
            out_scores = []
            with torch.no_grad():
                for pair in texts:
                    enc = self.tokenizer(pair[0], pair[1], return_tensors='pt', truncation=True, padding=True).to(next(self.model.parameters()).device)
                    logits = self.model(**enc).logits
                    score = torch.softmax(logits, dim=-1)[:, 1].cpu().item() if logits.shape[-1] > 1 else logits.cpu().item()
                    out_scores.append(float(score))
            return out_scores

        # Final fallback: heuristic diminishing scores
        return [float(1.0 / (1 + i)) for i in range(len(candidates))]

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores = self.score(query, candidates)
        for c, s in zip(candidates, scores):
            c['score'] = s
        return sorted(candidates, key=lambda x: x['score'], reverse=True)


if __name__ == "__main__":
    r = Reranker()
    cands = [{"id":"d1", "text": "Revenue increased 10%"},{"id":"d2", "text": "Expenses rose slightly"},{"id":"d3", "text": "Net income declined"}]
    print(r.rerank("What is the revenue trend?", cands))
