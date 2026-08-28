from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns float32 numpy array of shape (n, dim), L2-normalized for cosine similarity via inner product."""
    model = get_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vectors / norms

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
