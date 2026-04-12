"""Diagnose why p1_007 returns all NIH_ODS instead of USDA."""
from dotenv import load_dotenv
load_dotenv()

from zenic.rag.pipeline import generate_multi_queries, hybrid_search, rerank, _try_load_bm25_from_disk
from zenic.rag.vector_store import get_vector_store

_try_load_bm25_from_disk()

query = "plant based protein foods"

# 1. What multi-query variants are generated?
variants = generate_multi_queries(query, n=3)
print("VARIANTS:")
for v in variants: print(f"  {v}")

# 2. Raw hybrid search — what sources come up?
candidates = hybrid_search(variants, top_k=20)
print(f"\nHYBRID SEARCH top-20 sources:")
for c in candidates:
    print(f"  vec={c.get('vector_score',0):.3f} bm25={c.get('bm25_score',0):.3f} "
          f"src={c['metadata'].get('source')} :: {c['text'][:80].replace(chr(10),' ')}...")

# 3. After rerank
reranked = rerank(query, candidates, top_k=7)
print(f"\nRERANKED top-7:")
for c in reranked:
    print(f"  rerank={c.get('rerank_score',0):.3f} src={c['metadata'].get('source')} "
          f":: {c['text'][:80].replace(chr(10),' ')}...")

# 4. Direct USDA search — are plant foods actually indexed?
store = get_vector_store()
from zenic.rag.pipeline import _embed_model_instance
emb = _embed_model_instance().encode(query).tolist()
usda_results = store.search(emb, top_k=5, where={"source": "USDA"})
print(f"\nDIRECT USDA search (top-5):")
for r in usda_results:
    print(f"  score={r.get('vector_score',0):.3f} :: {r['text'][:100].replace(chr(10),' ')}...")
