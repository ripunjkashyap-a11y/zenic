"""Find DGA body text chunks that describe the Mediterranean diet pattern."""
import json, sys

sys.stdout.reconfigure(encoding="utf-8")
corpus = json.load(open("data/bm25_corpus.json", encoding="utf-8"))
dga = [c for c in corpus if c["metadata"].get("source") == "DietaryGuidelines"]

print(f"Total DGA chunks: {len(dga)}")
print()

# Search for Mediterranean-related content in any form
keywords = ["mediterranean", "seafood", "olive", "dietary pattern", "healthy pattern"]
for c in dga:
    t = c["text"].lower()
    matches = [k for k in keywords if k in t]
    if len(matches) >= 2:
        print(f"chunk {c['metadata']['chunk_index']} (matches: {matches}):")
        print(c["text"][:500])
        print("---")
