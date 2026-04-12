import json

corpus = json.load(open("data/bm25_corpus.json", encoding="utf-8"))

print("=== DGA chunks 138-143 ===")
dga = [c for c in corpus if c["metadata"].get("source") == "DietaryGuidelines"]
for c in dga:
    idx = c["metadata"]["chunk_index"]
    if 138 <= idx <= 143:
        print(f"\n--- chunk {idx} ---")
        print(c["text"][:600])

print("\n\n=== Vitamin D chunks 44-52 ===")
vitd = [c for c in corpus if c["metadata"].get("source") == "NIH_ODS"
        and "Vitamin D" in c["metadata"].get("nutrient_name", "")]
for c in vitd:
    idx = c["metadata"]["chunk_index"]
    if 44 <= idx <= 52:
        print(f"\n--- chunk {idx} ---")
        print(c["text"][:600])
