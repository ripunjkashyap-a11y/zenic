"""
Shared indexing logic — embeds documents and upserts them into the vector store.
All ingestion modules produce list[dict] with keys: id, text, metadata.
This module adds embeddings and persists to the store.
"""
import os
from typing import Callable
from sentence_transformers import SentenceTransformer

from zenic.rag.vector_store import get_vector_store

_embed_model: SentenceTransformer | None = None
_BATCH_SIZE = 64


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def index_documents(
    documents: list[dict],
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Embed and upsert a list of {id, text, metadata} documents.
    Returns the same list (for BM25 index building in the caller).
    Each document must have a unique `id`.
    """
    store = get_vector_store()
    model = _get_embed_model()
    total = len(documents)

    for start in range(0, total, _BATCH_SIZE):
        batch = documents[start : start + _BATCH_SIZE]
        texts = [d["text"] for d in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        enriched = [
            {**d, "embedding": emb}
            for d, emb in zip(batch, embeddings)
        ]
        store.upsert(enriched)
        if progress_cb:
            progress_cb(min(start + _BATCH_SIZE, total), total)

    return documents
