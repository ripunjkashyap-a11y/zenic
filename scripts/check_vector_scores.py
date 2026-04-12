"""Check vector scores for f2 and f5 problematic chunks — no LLM calls."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ.setdefault("GROQ_API_KEY", "dummy")

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from zenic.rag.pipeline import _try_load_bm25_from_disk, _embed_model_instance
from zenic.rag.vector_store import get_vector_store

_try_load_bm25_from_disk()
embedder = _embed_model_instance()
store = get_vector_store()

# --- f2: Does vitamin D UL chunk appear in top-30 vector results? ---
q2 = "What is the tolerable upper intake level for vitamin D in adults?"
e2 = embedder.encode(q2).tolist()
vr2 = store.search(query_embedding=e2, top_k=50)

print("=== f2: Vitamin D UL chunks in top-50 vector ===")
for r in vr2:
    t = r["text"]
    if "4,000 IU" in t or "Tolerable Upper Intake Level for vitamin D" in t:
        print(f"FOUND score={r['vector_score']:.3f} | {t[:150].replace(chr(10), ' ')}")

print("\n=== f2: Top-15 NIH_ODS by vector score ===")
nih = [r for r in vr2 if r["metadata"].get("source") == "NIH_ODS"]
for i, r in enumerate(nih[:15], 1):
    print(f"[{i}] {r['vector_score']:.3f} | {r['text'][:100].replace(chr(10), ' ')}")

# --- f5: Mediterranean chunk in top-50 vector? ---
q5 = "What does the Dietary Guidelines say about the Mediterranean diet?"
e5 = embedder.encode(q5).tolist()
vr5 = store.search(query_embedding=e5, top_k=50)

print("\n=== f5: Mediterranean chunk in top-50 vector ===")
found = False
for r in vr5:
    if "Mediterranean" in r["text"]:
        print(f"FOUND score={r['vector_score']:.3f} | {r['text'][:150].replace(chr(10), ' ')}")
        found = True
if not found:
    print("NOT FOUND in top-50")

print("\n=== f5: Top-10 DietaryGuidelines chunks by vector score ===")
dga = [r for r in vr5 if r["metadata"].get("source") == "DietaryGuidelines"]
for i, r in enumerate(dga[:10], 1):
    print(f"[{i}] {r['vector_score']:.3f} | {r['text'][:100].replace(chr(10), ' ')}")
