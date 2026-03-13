import os
from src.embed_and_store import Embedder
from src.rerank import Reranker


def test_embed_text_and_multimodal():
    e = Embedder()
    texts = ["Revenue grew 10% year-over-year", "Expenses declined"]
    emb = e.embed_text(texts)
    assert len(emb) == 2
    assert emb[0].dtype == 'float32'

    mm = e.embed_multimodal("Revenue up", None)
    assert mm.dtype == 'float32'


def test_reranker_scores():
    r = Reranker()
    cands = [{"text": "Revenue increased by 10%"}, {"text": "Costs increased"}]
    scores = r.score("What happened to revenue?", cands)
    assert len(scores) == 2
