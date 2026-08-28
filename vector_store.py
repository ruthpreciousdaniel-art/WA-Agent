import json
import faiss
import numpy as np
from app.config import INDEX_PATH, METADATA_PATH
from app.rag.embeddings import embed_texts

_index = None
_metadata: list[dict] = []  # aligned by faiss internal id order

def _dim() -> int:
    from app.rag.embeddings import get_embedder
    return get_embedder().get_sentence_embedding_dimension()

def _load():
    global _index, _metadata
    if _index is not None:
        return
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)
    else:
        _index = faiss.IndexFlatIP(_dim())  # inner product on normalized vecs = cosine sim
        _metadata = []

def _save():
    faiss.write_index(_index, str(INDEX_PATH))
    with open(METADATA_PATH, "w") as f:
        json.dump(_metadata, f)

def add_document(source_name: str, chunks: list[str]) -> int:
    """Embeds and adds chunks to the index. Returns number of chunks added."""
    _load()
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    _index.add(vectors)
    for chunk in chunks:
        _metadata.append({"source": source_name, "text": chunk})
    _save()
    return len(chunks)

def search(query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
    _load()
    if _index.ntotal == 0:
        return []
    query_vector = query_vector.reshape(1, -1)
    scores, ids = _index.search(query_vector, min(top_k, _index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = _metadata[idx]
        results.append({"text": item["text"], "source": item["source"], "score": float(score)})
    return results

def stats() -> dict:
    _load()
    sources = sorted(set(m["source"] for m in _metadata))
    return {"total_chunks": _index.ntotal, "sources": sources}
