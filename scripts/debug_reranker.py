"""
Check reranker scores directly for problematic chunks — no LLM calls.
Diagnoses why the reranker is misranking f2 and f5 chunks.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ.setdefault("GROQ_API_KEY", "dummy")

from dotenv import load_dotenv
load_dotenv()

from zenic.rag.pipeline import (
    _reranker_instance, _embed_model_instance, _try_load_bm25_from_disk, hybrid_search
)
import json

_try_load_bm25_from_disk()
reranker = _reranker_instance()
embedder = _embed_model_instance()

# --- f2 ---
q2 = "What is the tolerable upper intake level for vitamin D in adults?"
candidates2 = hybrid_search([q2], top_k=30, max_per_source=12)
pairs2 = [(q2, c["text"]) for c in candidates2]
scores2 = reranker.predict(pairs2)
ranked2 = sorted(zip(scores2, candidates2), reverse=True)

print("=== f2: Top-10 reranked candidates ===")
for score, c in ranked2[:10]:
    src = c["metadata"].get("source", "?")
    print(f"  {score:.3f} [{src}] {c['text'][:100].replace(chr(10), ' ')}")

# Check if vitamin D UL chunk appears
print("\n=== f2: Vitamin D UL chunks and their rerank scores ===")
for score, c in ranked2:
    t = c["text"]
    if "4,000 IU" in t or "Tolerable Upper Intake Level for vitamin D" in t:
        print(f"  {score:.3f} | {t[:150].replace(chr(10), ' ')}")

# --- f5 ---
q5 = "What does the Dietary Guidelines say about the Mediterranean diet?"
candidates5 = hybrid_search([q5], top_k=30, max_per_source=12)
pairs5 = [(q5, c["text"]) for c in candidates5]
scores5 = reranker.predict(pairs5)
ranked5 = sorted(zip(scores5, candidates5), reverse=True)

print("\n=== f5: Top-10 reranked candidates ===")
for score, c in ranked5[:10]:
    src = c["metadata"].get("source", "?")
    print(f"  {score:.3f} [{src}] {c['text'][:100].replace(chr(10), ' ')}")

print("\n=== f5: Mediterranean chunk rerank score ===")
for score, c in ranked5:
    if "Mediterranean" in c["text"]:
        print(f"  {score:.3f} | {c['text'][:150].replace(chr(10), ' ')}")

# --- f3 ---
q3 = "What does research say about protein requirements for resistance-trained athletes?"
candidates3 = hybrid_search([q3], top_k=30, max_per_source=12)
pairs3 = [(q3, c["text"]) for c in candidates3]
scores3 = reranker.predict(pairs3)
ranked3 = sorted(zip(scores3, candidates3), reverse=True)

print("\n=== f3: Top-10 reranked candidates ===")
for score, c in ranked3[:10]:
    src = c["metadata"].get("source", "?")
    print(f"  {score:.3f} [{src}] {c['text'][:100].replace(chr(10), ' ')}")

print("\n=== f3: ISSN chunks and their rerank scores ===")
for score, c in ranked3:
    if c["metadata"].get("source") == "ISSN":
        print(f"  {score:.3f} | {c['text'][:150].replace(chr(10), ' ')}")

# Check specifically for 1.6-2.2 g/kg content
print("\n=== f3: Chunks containing '1.6' or '2.2 g/kg' ===")
for score, c in ranked3:
    if "1.6" in c["text"] and ("2.2" in c["text"] or "g/kg" in c["text"].lower()):
        src = c["metadata"].get("source", "?")
        print(f"  {score:.3f} [{src}] {c['text'][:200].replace(chr(10), ' ')}")
