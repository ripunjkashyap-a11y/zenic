"""
Patch bm25_corpus.json in-place:
  1. Remove NIH ODS generic DRI boilerplate chunks (flood retrieval for nutrient queries).
  2. Add Mediterranean-Style Dietary Pattern description chunk (DGA table data was
     mangled by pypdf; this provides a BM25-retrievable summary of Table A3-5).
"""
import json
import re
import hashlib
from pathlib import Path

CORPUS_PATH = Path("data/bm25_corpus.json")

corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
original_count = len(corpus)

# --- 1. Remove NIH ODS boilerplate ---
BOILERPLATE = re.compile(
    r"^(Recommended Intakes\s+Intake recommendations for [^\n]+ are provided in the "
    r"Dietary Reference Intakes|Nutrient Intake Recommendations and Upper Limits)"
)

filtered = []
removed = 0
for chunk in corpus:
    if (
        chunk.get("metadata", {}).get("source") == "NIH_ODS"
        and BOILERPLATE.match(chunk.get("text", ""))
    ):
        removed += 1
    else:
        filtered.append(chunk)

print(f"Removed {removed} NIH ODS boilerplate chunks")

# --- 2. Add Mediterranean-Style Dietary Pattern chunk ---
med_text = (
    "Healthy Mediterranean-Style Dietary Pattern "
    "(Dietary Guidelines for Americans, 2020-2025, Appendix 3, Table A3-5)\n\n"
    "The Healthy Mediterranean-Style Dietary Pattern is adapted from traditional diets "
    "of countries bordering the Mediterranean Sea. It is one of three USDA Dietary Patterns "
    "in the Dietary Guidelines for Americans 2020-2025.\n\n"
    "Compared to the Healthy U.S.-Style Dietary Pattern, the Mediterranean-Style pattern "
    "includes:\n"
    "- More fruits (2.5 cup equivalents/day vs 2 cup eq/day)\n"
    "- More seafood (15 oz equivalents/week vs 8 oz eq/week) — fish such as salmon, sardines, "
    "trout, herring, and other seafood are emphasized\n"
    "- Less dairy (2 cup equivalents/day vs 3 cup eq/day)\n"
    "- Similar vegetables (2.5 cup equivalents/day), emphasizing dark-green, red and orange, "
    "beans, peas, and lentils\n"
    "- Oils including olive oil (27 grams/day)\n"
    "- Whole grains (at least half of 6 oz eq/day)\n"
    "- Nuts, seeds, and legumes\n\n"
    "The pattern limits added sugars, saturated fat, and sodium, consistent with all USDA "
    "Dietary Patterns. It does not prescribe specific foods but provides a flexible framework "
    "for customization."
)

chunk_id = hashlib.md5("DGA_mediterranean_style_A3-5".encode()).hexdigest()[:12]
med_chunk = {
    "id": f"dietary_{chunk_id}",
    "text": med_text,
    "metadata": {
        "source": "DietaryGuidelines",
        "source_name": "Dietary Guidelines for Americans 2020-2025",
        "category": "dietary_guidelines",
        "chunk_index": 9999,
        "note": "synthetic — Table A3-5 was not extractable by pypdf; re-ingest with improved parser to replace",
    },
}

# Only add if not already present
if not any(c.get("id") == med_chunk["id"] for c in filtered):
    filtered.append(med_chunk)
    print("Added Mediterranean-Style Dietary Pattern chunk")
else:
    print("Mediterranean chunk already present — skipped")

# --- Save ---
json.dump(filtered, open(CORPUS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=None)
print(f"Corpus: {original_count} → {len(filtered)} chunks (net {len(filtered) - original_count:+d})")
