"""
Verify the agent uses RAG for in-index queries and only falls back to live APIs
for genuinely absent items. Target: RAG hit rate >= 85%, 0 false API fallbacks.

Index ground-truth (confirmed via debug_scores.py, 2026-04-09):
  IN index  : chicken breast, egg, vitamin D (NIH_ODS), ISSN protein, barbell row
              (wger), cherimoya raw (USDA SR Legacy — exotic but indexed)
  NOT indexed: banana (only baby-food variants in 3k-chunk USDA subset), raw
               spinach (only baby-food spinach), white rice (not in USDA subset),
               jackfruit seeds (not in any source).
  Reranker scores (BAAI/bge-reranker-base): in-index >= 0.9, absent < 0.15.
  Threshold 0.5 cleanly separates the two groups.
"""
import json
from dotenv import load_dotenv
load_dotenv()
from zenic.agent.trace import run_with_trace

_CASES = [
    # --- Confirmed IN-INDEX (should use RAG, score >> 0.5) ---
    {"query": "protein in 100g chicken breast",            "should_use_rag": True},  # USDA
    {"query": "vitamin D upper intake level",              "should_use_rag": True},  # NIH ODS
    {"query": "ISSN protein recommendations for athletes", "should_use_rag": True},  # ISSN paper
    {"query": "barbell row muscles worked",                "should_use_rag": True},  # wger
    # --- Confirmed NOT IN INDEX (should trigger API fallback, score < 0.5) ---
    {"query": "calories in a medium banana",               "should_use_rag": False}, # absent from USDA subset
    {"query": "macros in boiled jackfruit seeds",          "should_use_rag": False}, # absent from all sources
]


def main():
    stats = {"total": 0, "correctly_used_rag": 0, "correctly_used_api": 0,
             "false_api_fallback": 0, "false_rag_attempt": 0, "failures": []}

    for case in _CASES:
        trace = run_with_trace(case["query"])
        tools = trace["tools_called"]
        used_rag = "rag_retrieval" in tools
        used_api = "usda_api" in tools or "wger_api" in tools
        stats["total"] += 1

        if case["should_use_rag"] and used_rag and not used_api:
            stats["correctly_used_rag"] += 1
        elif not case["should_use_rag"] and used_api:
            stats["correctly_used_api"] += 1
        elif case["should_use_rag"] and not used_rag:
            stats["false_api_fallback"] += 1
            stats["failures"].append({"query": case["query"], "expected": "rag", "actual": tools})
        else:
            stats["false_rag_attempt"] += 1
            stats["failures"].append({"query": case["query"], "expected": "api", "actual": tools})

    in_scope = sum(1 for c in _CASES if c["should_use_rag"])
    rag_rate = stats["correctly_used_rag"] / in_scope if in_scope else 0

    print(json.dumps(stats, indent=2))
    print(f"\nRAG HIT RATE (in-scope queries): {rag_rate:.0%}")
    print(f"FALSE API FALLBACKS:             {stats['false_api_fallback']}")

    if rag_rate >= 0.85 and stats["false_api_fallback"] == 0:
        print("\nGREEN -- Ready to proceed to Pillar 2")
    elif rag_rate >= 0.70:
        print("\nYELLOW -- Index has gaps; check inspect_chunks.py")
    else:
        print("\nRED -- RAG is not functioning as primary retrieval path")


if __name__ == "__main__":
    main()
