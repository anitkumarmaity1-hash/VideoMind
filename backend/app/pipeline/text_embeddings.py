"""
Text embedding generation using Sentence Transformers (default: BGE small).
"""
from typing import List
from app.config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.text_embedding_model)
    return _model


def embed_texts(texts: List[str], normalize: bool = True) -> List[List[float]]:
    """Embed a batch of texts. Empty strings are embedded as-is (zero-ish vector)."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(text: str, normalize: bool = True) -> List[float]:
    return embed_texts([text], normalize=normalize)[0]
