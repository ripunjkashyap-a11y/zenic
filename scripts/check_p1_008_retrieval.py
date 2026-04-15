"""Inspect retrieved chunks for p1_008 to diagnose faithfulness=0.000."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from zenic.rag.pipeline import hybrid_search, rerank, generate, _try_load_bm25_from_disk

_try_load_bm25_from_disk()
q = "how much calcium do pregnant women need"
candidates = hybrid_search([q], top_k=30, max_per_source=12)
chunks = rerank(q, candidates, top_k=9)

print(f"\nTop-9 chunks for: '{q}'\n" + "=" * 70)
for i, c in enumerate(chunks):
    src = c.get("metadata", {}).get("source", "?")
    score = c.get("rerank_score", 0)
    print(f"\n{i+1}. [{score:.3f}] [{src}]\n   {c['text'][:300]}")

print("\n" + "=" * 70)
print("Generated answer:")
print(generate(q, chunks))
