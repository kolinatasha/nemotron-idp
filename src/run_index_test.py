"""Simple test runner to validate Milvus indexing and search.

Creates a collection, inserts random vectors with metadata, and performs a search.
"""
import numpy as np
from src.embed_and_store import index_documents, search_milvus


def main():
    milvus_cfg = {"uri": "tcp://127.0.0.1:19530", "alias": "default"}
    collection = "test_docs"

    # Create dummy vectors and metadata
    num = 10
    dim = 2048
    vectors = [np.random.rand(dim).astype('float32') for _ in range(num)]
    metadatas = [{"source": f"doc_{i}", "text": f"dummy text {i}"} for i in range(num)]

    print("Indexing vectors into Milvus (may take a moment)...")
    res = index_documents(milvus_cfg, vectors, metadatas, collection=collection)
    print("Insert result:", res)

    # Query with one of the vectors
    qv = vectors[0]
    print("Searching for similar vectors...")
    hits = search_milvus(milvus_cfg, collection, qv, top_k=5)
    print("Hits:")
    for h in hits:
        print(h)


if __name__ == "__main__":
    main()
