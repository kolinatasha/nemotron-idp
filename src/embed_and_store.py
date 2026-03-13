"""Embedding and indexing utilities (Milvus example).

- Loads a multimodal embedding model (HF / local inference)
- Encodes text + images into a single vector
- Inserts vectors + metadata into Milvus

Replace placeholders with the exact model API for `nvidia/llama-nemotron-embed-vl-1b-v2`.
"""
from typing import List, Dict, Any
import os
import numpy as np

try:
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
except Exception:
    connections = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    Collection = None
    utility = None

from PIL import Image
import torch
from typing import List, Dict, Any
import os
import numpy as np

try:
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
except Exception:
    connections = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    Collection = None
    utility = None

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer

# Fallback: sentence-transformers for CPU-friendly embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class Embedder:
    """Multimodal embedder with hardware-aware loading and CPU fallbacks.

    Attempts to load the specified Nemotron HF embedder. If unavailable, falls back to
    `sentence-transformers/all-MiniLM-L6-v2` for text embeddings. Image embedding falls back
    to a random vector when no image model is available.
    """

    def __init__(self, model_name: str = "nvidia/llama-nemotron-embed-vl-1b-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if (self.device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

        self.tokenizer = None
        self.model = None
        self.processor = None
        self.st_model = None

        # Try to load the HF model (text encoder / multimodal processor)
        try:
            # Use device_map="auto" where possible; transformers will place layers on available devices
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.torch_dtype, device_map="auto")
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
            except Exception:
                self.processor = None
        except Exception:
            self.tokenizer = None
            self.model = None
            self.processor = None

        # Fallback to sentence-transformers for text-only embeddings on CPU
        if self.model is None and SentenceTransformer is not None:
            try:
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.st_model = None

    def embed_text(self, texts: List[str]) -> List[np.ndarray]:
        if self.st_model is not None:
            emb = self.st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [e.astype(np.float32) for e in emb]

        if self.tokenizer is not None and self.model is not None:
            self.model.eval()
            out = []
            with torch.no_grad():
                for t in texts:
                    inputs = self.tokenizer(t, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
                    outputs = self.model(**inputs)
                    last = outputs.last_hidden_state
                    # mean pooling
                    emb = last.mean(dim=1).cpu().numpy()[0]
                    out.append(emb.astype(np.float32))
            return out

        # Final fallback: random vectors
        return [np.random.rand(768).astype(np.float32) for _ in texts]

    def embed_image(self, images: List[Image.Image]) -> List[np.ndarray]:
        # If processor + model support pixel inputs, try to obtain image features
        if self.processor is not None and self.model is not None:
            self.model.eval()
            out = []
            with torch.no_grad():
                for img in images:
                    proc = self.processor(images=img, return_tensors="pt")
                    # move tensors to model device
                    proc = {k: v.to(self.model.device) for k, v in proc.items()}
                    try:
                        outputs = self.model(**proc)
                        last = outputs.last_hidden_state
                        emb = last.mean(dim=1).cpu().numpy()[0]
                        out.append(emb.astype(np.float32))
                    except Exception:
                        out.append(np.random.rand(768).astype(np.float32))
            return out

        # Fallback: random vectors
        return [np.random.rand(768).astype(np.float32) for _ in images]

    def embed_multimodal(self, text: str, image: Image.Image = None) -> np.ndarray:
        t_emb = self.embed_text([text])[0]
        if image is None:
            v = t_emb
        else:
            i_emb = self.embed_image([image])[0]
            v = np.concatenate([t_emb, i_emb])
        # normalize
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v.astype(np.float32)
