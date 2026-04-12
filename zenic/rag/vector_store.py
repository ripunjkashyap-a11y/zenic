"""
Vector store factory — switches between ChromaDB (local dev) and Qdrant Cloud (production)
based on the ENV environment variable.
"""
import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, documents: list[dict]) -> None: ...
    def search(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]: ...
    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]: ...
    def delete_by_source(self, source: str) -> int: ...


def get_vector_store() -> VectorStore:
    env = os.getenv("ENV", "development")
    if env == "production":
        return _get_qdrant_store()
    return _get_chroma_store()


def _get_chroma_store() -> VectorStore:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    return _ChromaAdapter(client)


def _get_qdrant_store() -> VectorStore:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    return _QdrantAdapter(client)


class _ChromaAdapter:
    COLLECTION = "zenic_knowledge"

    def __init__(self, client):
        self._col = client.get_or_create_collection(self.COLLECTION)

    def upsert(self, documents: list[dict]) -> None:
        self._col.upsert(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            embeddings=[d["embedding"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
        )

    def search(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        results = self._col.query(**kwargs)
        return [
            {"text": doc, "metadata": meta, "vector_score": 1 - dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]:
        kwargs = {"limit": n}
        if where:
            kwargs["where"] = where
        results = self._col.get(**kwargs)
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def delete_by_source(self, source: str) -> int:
        results = self._col.get(where={"source": source}, include=[])
        ids = results["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)


class _QdrantAdapter:
    COLLECTION = "zenic_knowledge"

    def __init__(self, client):
        self._client = client

    def upsert(self, documents: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=d["id"], vector=d["embedding"], payload={**d["metadata"], "text": d["text"]})
            for d in documents
        ]
        self._client.upsert(collection_name=self.COLLECTION, points=points)

    def search(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = None
        if where:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
            query_filter = Filter(must=conditions)
        results = self._client.search(
            collection_name=self.COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            {"text": r.payload["text"], "metadata": {k: v for k, v in r.payload.items() if k != "text"}, "vector_score": r.score}
            for r in results
        ]

    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]:
        results, _ = self._client.scroll(
            collection_name=self.COLLECTION,
            limit=n,
        )
        return [
            {"text": r.payload["text"], "metadata": {k: v for k, v in r.payload.items() if k != "text"}}
            for r in results
        ]

    def delete_by_source(self, source: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
        result = self._client.delete(collection_name=self.COLLECTION, points_selector=query_filter)
        return result.operation_id or 0
