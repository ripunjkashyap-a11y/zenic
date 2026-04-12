"""
Wrapper node — delegates to the Pillar 1 pipeline.

RAG-first policy: always try the local index first. If the top reranked
result scores below FALLBACK_THRESHOLD (meaning the index doesn't have a
good answer), fall back to the live USDA FoodData Central API and record
the fallback so trace.py / rag_vs_api_check.py can detect it.
"""
import os

from zenic.agent.state import ZenicState
from zenic.rag.pipeline import retrieve

# Threshold for "index returned nothing useful". BAAI/bge-reranker-base
# outputs raw logits; in-index hits typically score > 0, genuine misses < -1.
# Tune this value against retrieval_spot_check output if needed.
_FALLBACK_SCORE_THRESHOLD = 0.5


def _poor_retrieval(chunks: list[dict]) -> bool:
    """True when RAG found nothing relevant for the query."""
    if not chunks:
        return True
    return chunks[0].get("rerank_score", 0.0) < _FALLBACK_SCORE_THRESHOLD


def run(state: ZenicState) -> dict:
    query = state["messages"][-1].content
    chunks = retrieve(query)

    if _poor_retrieval(chunks) and os.environ.get("USDA_API_KEY"):
        from zenic.agent.tools import usda_api
        try:
            api_results = usda_api.search_food(query)
            api_chunks = [
                {
                    "text": (
                        f"{item['name']}\n"
                        + "\n".join(f"  {k}: {v}" for k, v in item.get("nutrients", {}).items())
                    ),
                    "metadata": {"source": "USDA FoodData Central (live)", "year": "2024"},
                }
                for item in api_results
            ]
            if api_chunks:
                return {
                    "retrieved_context": api_chunks,
                    "tool_results": {"api_fallback_used": "usda_api"},
                }
        except Exception:
            # Live API unavailable — fall through and return whatever RAG found
            pass

    return {"retrieved_context": chunks}
