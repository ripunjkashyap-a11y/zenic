# Zenic V1 — Pillar 1: Testing & Spot-Check Guide

> **Purpose:** Lightweight manual and spot-check testing to run *during* RAG development — before Pillar 3's full evaluation pipeline exists. The goal is to catch retrieval and chunking issues early, while they're cheap to fix, instead of discovering them buried under Pillar 2's agent logic.
>
> **This is NOT the full eval pipeline.** That lives in Pillar 3. This is the fast feedback loop you run after every chunking change, every embedding swap, every reranker tweak. Eyeball-driven, not metric-driven.

---

## 1. When to Use This Guide

Run these checks whenever you:

- Change the chunking strategy for any source
- Swap embedding models (`all-MiniLM-L6-v2` ↔ `BGE-small-en-v1.5`)
- Add or remove a data source from the index
- Tune the hybrid search weights (vector vs BM25)
- Swap or configure the cross-encoder reranker
- Change multi-query generation prompts
- Before handing the RAG off to Pillar 2's agent layer

**Time budget:** 10–20 minutes per run. If it's taking longer, automate the painful parts or wait for Pillar 3.

---

## 2. Mini Eval Dataset (12 Examples)

Hand-crafted, deliberately small. Each example targets a specific retrieval failure mode. If any of these 12 break, something is wrong — don't advance to Pillar 2 until they pass.

Store this as `eval_data/pillar1_spot_check.json`.

```json
[
  {
    "id": "p1_001",
    "category": "usda_food_basic",
    "query": "How much protein is in 100g of cooked chicken breast?",
    "expected_source": "USDA FoodData Central",
    "expected_keywords_in_context": ["chicken breast", "protein", "31"],
    "expected_answer_contains": ["31", "protein", "chicken"],
    "failure_mode_tested": "Basic vector retrieval on common food",
    "notes": "Sanity check. If this fails, indexing is broken."
  },
  {
    "id": "p1_002",
    "category": "usda_food_bm25",
    "query": "iron content in raw spinach",
    "expected_source": "USDA FoodData Central",
    "expected_keywords_in_context": ["spinach", "iron", "2.7"],
    "expected_answer_contains": ["2.7", "iron", "spinach"],
    "failure_mode_tested": "BM25 exact-term matching",
    "notes": "Specifically designed to stress BM25. If hybrid search is broken or BM25 weight is zero, this will return generic 'leafy greens' content instead of exact spinach data."
  },
  {
    "id": "p1_003",
    "category": "nih_rda",
    "query": "What is the tolerable upper intake level for vitamin D in adults?",
    "expected_source": "NIH Office of Dietary Supplements",
    "expected_keywords_in_context": ["vitamin D", "upper intake", "4000", "IU"],
    "expected_answer_contains": ["4000 IU", "100 mcg", "adults"],
    "failure_mode_tested": "NIH fact sheet retrieval, specific numeric value",
    "notes": "Tests that supplement fact sheets are chunked so the UL isn't separated from its context."
  },
  {
    "id": "p1_004",
    "category": "issn_paper_section_aware",
    "query": "protein requirements for resistance trained athletes",
    "expected_source": "ISSN Position Stand on Protein and Exercise",
    "expected_keywords_in_context": ["1.6", "2.2", "kg", "protein", "resistance"],
    "expected_answer_contains": ["1.6", "2.2", "per kilogram"],
    "failure_mode_tested": "Section-aware chunking on long scientific document",
    "notes": "If the retrieved chunk is missing the author/year/section metadata prefix, section-aware chunking is broken."
  },
  {
    "id": "p1_005",
    "category": "wger_exercise",
    "query": "what are good compound exercises for back using a barbell",
    "expected_source": "wger",
    "expected_keywords_in_context": ["barbell row", "deadlift", "back"],
    "expected_answer_contains": ["row", "deadlift"],
    "failure_mode_tested": "Exercise DB retrieval with metadata filter intent",
    "notes": "Tests whether exercise metadata (muscle_group, equipment) is being used in retrieval."
  },
  {
    "id": "p1_006",
    "category": "dietary_guidelines",
    "query": "what does the Mediterranean diet typically include",
    "expected_source": "Dietary Guidelines",
    "expected_keywords_in_context": ["olive oil", "fish", "vegetables", "Mediterranean"],
    "expected_answer_contains": ["olive oil", "fish", "vegetables"],
    "failure_mode_tested": "Long-form dietary guidelines retrieval with recursive chunking",
    "notes": "If chunk overlap is wrong, may return a fragment that lists only one food group."
  },
  {
    "id": "p1_007",
    "category": "multi_query_expansion",
    "query": "plant based protein foods",
    "expected_source": "USDA FoodData Central",
    "expected_keywords_in_context": ["lentils", "tofu", "beans", "protein"],
    "expected_answer_contains": ["lentils", "tofu"],
    "failure_mode_tested": "Multi-query generation — 'plant based' should expand to 'vegan', 'vegetarian', 'plant protein sources'",
    "notes": "If multi-query is disabled, this may only match documents containing the exact phrase 'plant based'."
  },
  {
    "id": "p1_008",
    "category": "reranker_disambiguation",
    "query": "how much calcium do pregnant women need",
    "expected_source": "NIH Office of Dietary Supplements",
    "expected_keywords_in_context": ["calcium", "pregnancy", "1000", "mg"],
    "expected_answer_contains": ["1000 mg", "pregnancy"],
    "failure_mode_tested": "Cross-encoder reranking on life-stage-specific RDA",
    "notes": "Initial retrieval may return generic calcium RDA. Reranker should promote the pregnancy-specific chunk. Compare pre- and post-rerank lists."
  },
  {
    "id": "p1_009",
    "category": "api_fallback_triggering",
    "query": "nutritional info for raw cherimoya",
    "expected_source": "USDA API (live)",
    "expected_keywords_in_context": ["cherimoya"],
    "expected_answer_contains": ["cherimoya"],
    "failure_mode_tested": "API fallback when item not in local index",
    "notes": "IMPORTANT — verify this item is absent from the local index before trusting the result. Query the index directly first; if cherimoya is present, the test will silently pass via RAG and not exercise API fallback at all. Swap for another genuinely uncommon item (e.g. soursop pulp, feijoa, jackfruit seeds boiled) if needed. Durian was the original choice here but it's a major commercial fruit in Southeast Asia and likely sits in any top-2000 USDA cut; ugli fruit has the same problem (USDA FDC indexes it as 'Tangelo, Uniq fruit'). Prefer items that are structurally absent (specific preparations, regional dishes) over items that are merely statistically obscure."
  },
  {
    "id": "p1_010",
    "category": "rag_first_enforcement",
    "query": "calories in a medium banana",
    "expected_source": "USDA FoodData Central (local index)",
    "expected_keywords_in_context": ["banana", "calories", "105"],
    "expected_answer_contains": ["105", "calories"],
    "failure_mode_tested": "Agent does NOT call USDA API when local index has the answer",
    "notes": "Common food, must be in local index. If the agent calls the API anyway, RAG-first pattern is broken."
  },
  {
    "id": "p1_011",
    "category": "citation_presence",
    "query": "what does research say about creatine loading phases",
    "expected_source": "ISSN Position Stand on Creatine",
    "expected_keywords_in_context": ["creatine", "loading", "20", "grams"],
    "expected_answer_contains": ["ISSN", "loading", "creatine"],
    "failure_mode_tested": "Source citation formatting in LLM output",
    "notes": "Answer MUST cite source by name and year. If citation is missing, system prompt or metadata passthrough is broken."
  },
  {
    "id": "p1_012",
    "category": "no_answer_handling",
    "query": "what color is the vitamin D molecule",
    "expected_source": null,
    "expected_keywords_in_context": [],
    "expected_answer_contains": ["don't have", "not sure", "cannot"],
    "failure_mode_tested": "Graceful handling when retrieval returns irrelevant content",
    "notes": "Trick query. The index has vitamin D nutrition info but nothing about molecular color. LLM must not confabulate — should say it doesn't know."
  }
]
```

**Why exactly these 12?** Each one targets a specific component:

| # | Component being stressed |
|---|---|
| 001 | Vector index basic sanity |
| 002 | BM25 / hybrid search |
| 003 | NIH chunking |
| 004 | ISSN section-aware chunking + metadata |
| 005 | wger exercise indexing |
| 006 | Recursive chunking on long docs |
| 007 | Multi-query expansion |
| 008 | Cross-encoder reranker |
| 009 | API fallback triggers correctly |
| 010 | RAG-first enforcement (API does NOT trigger) |
| 011 | Source citation in generation |
| 012 | Graceful failure / anti-hallucination |

If you skip an example, you lose coverage of that component.

---

## 3. Chunking Quality — Manual Inspection

Before you retrieve anything, verify the chunks themselves look sane. Bad chunks compound into every downstream problem.

### 3.1 The Chunk Inspection Script

```python
# scripts/inspect_chunks.py
import random
from zenic.rag.vector_store import get_vector_store

def inspect_chunks(source_filter: str, n: int = 10):
    """Pull n random chunks from a given source and print them."""
    store = get_vector_store()
    chunks = store.sample_chunks(where={"source": source_filter}, n=n)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'=' * 70}")
        print(f"CHUNK {i} / {n}  |  source={source_filter}")
        print(f"{'=' * 70}")
        print(f"METADATA: {chunk.metadata}")
        print(f"LENGTH:   {len(chunk.text)} chars, ~{len(chunk.text.split())} words")
        print(f"{'-' * 70}")
        print(chunk.text)

if __name__ == "__main__":
    for source in ["USDA", "NIH_ODS", "ISSN", "wger", "DietaryGuidelines"]:
        inspect_chunks(source, n=5)
```

### 3.2 What to Look For

Run the script and eyeball the output against this checklist:

**USDA food chunks:**
- Each chunk should contain ONE food item with its full nutrient profile
- Metadata must include `category`, `food_group`, `source="USDA"`
- No truncated nutrient lists (did the chunker cut off mid-table?)

**NIH supplement chunks:**
- Each chunk is self-contained — the nutrient name appears inside the chunk, not just in metadata
- Numeric values (RDA, UL) are not separated from their units or their life-stage context
- Red flag: a chunk that says "4000 IU" without saying "vitamin D" or "adults"

**ISSN scientific paper chunks:**
- Each chunk MUST start with a metadata prefix like:
  `"Source: ISSN Position Stand on Protein and Exercise (Jäger et al., 2017), Section: Recommendations"`
- Chunks should land on section boundaries, not mid-sentence
- Chunk length roughly 800–1000 tokens — if they're all 200 tokens, section-aware chunking is broken
- Overlap should preserve context between adjacent chunks

**wger exercise chunks:**
- One exercise per chunk with name, description, muscle group, equipment
- Metadata tags populated correctly

**Dietary guidelines chunks:**
- Chunks should cohere around a single topic (a diet type, a macronutrient, a population group)
- If a chunk mixes "vegan macros" with "pregnancy iron needs", overlap is too aggressive or splitter is wrong

### 3.3 Quick Smell Tests

Five 10-second checks to run after any chunking change:

1. **Word count distribution.** Plot a histogram of chunk word counts per source. Each source should show a tight distribution near its target size. Bimodal or extremely spread out = problem.
2. **Orphan chunks.** Any chunk under 50 words is probably garbage from a bad split. Count them; target is < 2% of total.
3. **Metadata completeness.** For each source, check that 100% of chunks have all required metadata fields. A missing `source_title` on an ISSN chunk means citations will fail at generation time.
4. **Duplicate detection.** Hash each chunk's text. If you have > 5% duplicates, your ingestion is double-indexing.
5. **Encoding check.** Grep for `�` or `\x` in a sample. Encoding issues in scientific PDFs are common and silently poison retrieval.

---

## 4. Retrieval Spot-Check Script

The core workflow: run a query, print what came back at every stage, eyeball it.

### 4.1 The Script

```python
# scripts/retrieval_spot_check.py
from zenic.rag.pipeline import (
    generate_multi_queries,
    hybrid_search,
    rerank,
)

def spot_check(query: str, verbose: bool = True):
    print(f"\n{'#' * 70}")
    print(f"QUERY: {query}")
    print(f"{'#' * 70}")

    # Stage 1: Multi-query expansion
    variants = generate_multi_queries(query)
    print(f"\n[1] MULTI-QUERY VARIANTS ({len(variants)}):")
    for v in variants:
        print(f"    - {v}")

    # Stage 2: Hybrid search (vector + BM25)
    candidates = hybrid_search(variants, top_k=20)
    print(f"\n[2] HYBRID SEARCH — TOP 20 CANDIDATES:")
    for i, c in enumerate(candidates, 1):
        snippet = c.text[:120].replace("\n", " ")
        print(f"  {i:>2}. [vec={c.vector_score:.3f} bm25={c.bm25_score:.3f}] "
              f"src={c.metadata.get('source')} :: {snippet}...")

    # Stage 3: Rerank
    reranked = rerank(query, candidates, top_k=7)
    print(f"\n[3] AFTER CROSS-ENCODER RERANK — TOP 7:")
    for i, c in enumerate(reranked, 1):
        snippet = c.text[:120].replace("\n", " ")
        print(f"  {i}. [rerank={c.rerank_score:.3f}] "
              f"src={c.metadata.get('source')} :: {snippet}...")

    if verbose:
        print(f"\n[4] FULL TEXT OF TOP-3 RERANKED CHUNKS:")
        for i, c in enumerate(reranked[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Metadata: {c.metadata}")
            print(c.text)

if __name__ == "__main__":
    import json
    with open("eval_data/pillar1_spot_check.json") as f:
        cases = json.load(f)
    for case in cases:
        spot_check(case["query"])
        input("\n[Press Enter for next query...]")
```

### 4.2 How to Read the Output

For each query, ask yourself these questions:

1. **Are the multi-query variants sensible?** Do they actually rephrase the query, or are they near-duplicates? If all four variants are basically the same sentence, the multi-query prompt needs work.
2. **Does the top candidate from hybrid search look relevant?** If not, check whether the vector score or BM25 score is dominating — one might be starving the other.
3. **Did the reranker change the order?** Compare pre-rerank top 5 vs post-rerank top 5. If the reranker isn't moving anything, it's not pulling its weight.
4. **Are all expected sources represented?** If a query about "protein for athletes" returns zero ISSN chunks, your scientific papers aren't indexed properly or the embeddings don't match the query phrasing.
5. **Is the reranker promoting noise?** Occasionally the cross-encoder promotes a bad match with high confidence. Catch these early.

### 4.3 Before/After Comparison Mode

When you change something (new embedding model, new reranker, new chunk size), run the same query set with the old config saved and diff the results:

```python
# Save baseline
python scripts/retrieval_spot_check.py --save baseline_v1.json

# ... make your change ...

python scripts/retrieval_spot_check.py --compare baseline_v1.json
```

The comparison output should flag: (a) queries where the top-3 chunks changed completely, (b) queries where the new top-1 is a chunk that wasn't even in the old top-20, (c) queries where the old top-1 dropped out entirely.

This is your cheap A/B test. It's not statistically rigorous — that's Pillar 3's job — but it catches regressions fast.

---

## 5. Faithfulness Spot-Check (No RAGAS Required)

Full faithfulness scoring with RAGAS comes in Pillar 3. For dev-time spot-checking, a manual side-by-side is enough.

### 5.1 The Manual Check

```python
# scripts/faithfulness_spot_check.py
from zenic.rag.pipeline import retrieve, generate

def faithfulness_check(query: str):
    chunks = retrieve(query)
    answer = generate(query, chunks)

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}")

    print(f"\n>>> RETRIEVED CONTEXT ({len(chunks)} chunks):")
    for i, c in enumerate(chunks, 1):
        print(f"\n[Chunk {i}] source={c.metadata.get('source')}")
        print(c.text)

    print(f"\n>>> GENERATED ANSWER:")
    print(answer)

    print(f"\n>>> MANUAL CHECKLIST:")
    print("  [ ] Every factual claim in the answer appears in at least one chunk")
    print("  [ ] No numeric values in the answer are absent from the chunks")
    print("  [ ] Sources are cited by name and year")
    print("  [ ] No confabulation (claims not traceable to context)")
    print("  [ ] Caveats and safety language present where appropriate")
```

### 5.2 What Counts as a Faithfulness Failure

You're looking for three specific failure patterns:

1. **Fabricated numbers.** The LLM says "contains 28g of protein" but the retrieved chunk says "31g". This is the most dangerous failure mode in a health app. Zero tolerance.
2. **Fabricated sources.** The LLM cites "according to a 2019 Harvard study" when the retrieved context contains no such study. Check every citation against metadata.
3. **Extrapolation beyond context.** The chunks discuss creatine for strength training, the LLM extends that to recommending creatine for endurance athletes. The leap may or may not be factually correct, but it's not *grounded* — and that's what faithfulness measures.

Log every failure with: query, retrieved chunks, generated answer, which failure mode. These become regression tests for Pillar 3.

### 5.3 The Five-Query Daily Smoke Test

Pick five queries from the mini eval dataset and run them at the start of every dev session. Paste the output into a scratch file, scan it for 60 seconds, move on. This is your early warning system — if today's output looks different from yesterday's, *something* changed.

Recommended set: `p1_001, p1_004, p1_008, p1_010, p1_012`. Covers basic retrieval, section-aware chunking, reranking, API fallback enforcement, and anti-hallucination.

---

## 6. RAG-First / API-Fallback Spot-Check

The core question: is the agent actually using the RAG, or quietly bypassing it by calling the USDA and wger APIs?

If it over-relies on APIs, the entire Pillar 1 effort is wasted. This check is non-negotiable.

### 6.1 The Trace-and-Count Script

```python
# scripts/rag_vs_api_check.py
import json
from zenic.agent.trace import run_with_trace

def rag_usage_check(test_cases: list) -> dict:
    stats = {
        "total": 0,
        "correctly_used_rag": 0,
        "correctly_used_api": 0,
        "false_api_fallback": 0,  # Should have used RAG, used API instead
        "false_rag_attempt": 0,   # Should have used API, used RAG instead
        "failures": []
    }

    for case in test_cases:
        trace = run_with_trace(case["query"])
        tools_called = trace["tools_called"]
        used_rag = "rag_search" in tools_called
        used_api = "usda_api" in tools_called or "wger_api" in tools_called

        stats["total"] += 1
        should_rag = case["should_use_rag"]

        if should_rag and used_rag and not used_api:
            stats["correctly_used_rag"] += 1
        elif not should_rag and used_api:
            stats["correctly_used_api"] += 1
        elif should_rag and not used_rag:
            stats["false_api_fallback"] += 1
            stats["failures"].append({
                "query": case["query"],
                "expected": "rag",
                "actual_tools": tools_called,
                "trace_id": trace["id"]
            })
        else:
            stats["false_rag_attempt"] += 1
            stats["failures"].append({
                "query": case["query"],
                "expected": "api",
                "actual_tools": tools_called,
                "trace_id": trace["id"]
            })

    return stats

if __name__ == "__main__":
    # Minimal 10-case set for dev-time spot-checking
    cases = [
        {"query": "calories in a medium banana", "should_use_rag": True},
        {"query": "protein in 100g chicken breast", "should_use_rag": True},
        {"query": "iron content of raw spinach", "should_use_rag": True},
        {"query": "macros in 100g white rice", "should_use_rag": True},
        {"query": "calories in a large egg", "should_use_rag": True},
        {"query": "vitamin D upper intake level", "should_use_rag": True},
        {"query": "ISSN protein recommendations for athletes", "should_use_rag": True},
        {"query": "barbell row muscles worked", "should_use_rag": True},
        {"query": "nutritional info for raw cherimoya", "should_use_rag": False},
        {"query": "macros in boiled jackfruit seeds", "should_use_rag": False},
    ]
    stats = rag_usage_check(cases)
    print(json.dumps(stats, indent=2))

    rag_rate = stats["correctly_used_rag"] / max(
        sum(1 for c in cases if c["should_use_rag"]), 1
    )
    print(f"\nRAG HIT RATE ON IN-SCOPE QUERIES: {rag_rate:.0%}")
    print(f"FALSE API FALLBACKS: {stats['false_api_fallback']}")
```

### 6.2 Interpreting the Results

**Green light:** RAG hit rate ≥ 85% on in-scope queries AND 0 false API fallbacks on the sanity cases (`banana`, `chicken breast`). Proceed to Pillar 2.

**Yellow light:** RAG hit rate 70–85%. The index has gaps. Before moving on, run `inspect_chunks.py` and check whether the failed items are actually in the index. If they are, the retriever's scoring threshold is too aggressive — loosen it.

**Red light:** RAG hit rate < 70%, OR any of the sanity cases (banana, chicken, egg, rice) falls back to the API. Stop. The RAG is not functioning as the primary retrieval path. Likely causes:

- Agent's tool-choice system prompt is biasing toward API calls
- Retriever is returning low confidence scores even on good matches (embedding model mismatch?)
- Top-k is too restrictive and the match is getting filtered out
- Indexing silently failed for those food items (check chunk inspection)

### 6.3 Why This Check Exists Separately

You could fold this into the main retrieval spot-check, but keeping it separate has a purpose: it's the one test that validates the *architectural* decision behind Pillar 1. Everything else in this guide checks whether retrieval works. This one checks whether retrieval is *being used*. They're different failures with different fixes.

---

## 7. Daily Dev Workflow

Put it all together. This is what your dev loop looks like while building Pillar 1:

```
Morning:
  1. Five-query smoke test (Section 5.3) — 2 min
  2. Glance at yesterday's chunk inspection output — 1 min

After any chunking change:
  1. Re-run inspect_chunks.py on the affected source — 3 min
  2. Re-run retrieval_spot_check.py on relevant queries — 5 min
  3. Smell-test the word count distribution — 1 min

After any retrieval pipeline change (embeddings, reranker, hybrid weights):
  1. Save baseline before change
  2. Make change
  3. Run retrieval_spot_check.py in compare mode — 5 min
  4. Eyeball the diff for regressions — 3 min

Before declaring Pillar 1 done:
  1. Run all 12 mini eval cases, each must pass eyeball test — 15 min
  2. Run rag_vs_api_check.py, must be green — 5 min
  3. Run faithfulness spot-check on 5 random queries — 10 min
  4. Document results in dev log
  5. Hand off to Pillar 2
```

---

## 8. What This Guide Deliberately Does NOT Cover

To avoid overlapping with Pillar 3, these are explicitly out of scope:

- **RAGAS metrics.** No faithfulness/relevance/precision/recall numbers here. Spot-checks only.
- **LLM-as-a-Judge scoring.** Pillar 3's Gemini judge is for final eval, not dev iteration.
- **Router accuracy.** That's a Pillar 2 concern — the router doesn't exist yet at this stage.
- **Safety block testing.** Pillar 3 has a dedicated safety test suite. Here, you only check that the RAG doesn't hallucinate — not that it blocks harmful requests.
- **Statistical significance.** 12 examples is not a statistically meaningful eval. It's a sanity check. Don't over-claim.
- **Continuous tracking over time.** Pillar 3's dashboard handles historical comparison. Here, you just eyeball the current run.

If you find yourself wanting any of the above during Pillar 1 development, that's a signal it's time to move on to Pillar 2 and eventually Pillar 3 — not to bloat this guide.

---

## 9. Dev Log Template

After each spot-check session, log results. One-minute effort, pays off later when you're writing the README.

```markdown
## Spot-Check Log — 2026-04-06

**Change made:** Switched embedding model from all-MiniLM-L6-v2 to BGE-small-en-v1.5

**Mini eval results (12 cases):**
- Passed eyeball: 11/12
- Failed: p1_008 (calcium for pregnancy) — reranker still not promoting the pregnancy-specific chunk

**RAG vs API check:** 9/10 correct (1 false API fallback on "macros in 100g white rice")
  - Cause: rice not in top 2000 USDA index. Fix: expand index.

**Chunk inspection:** word count distributions look clean across all sources

**Decision:** BGE-small is clearly better on scientific queries (p1_004, p1_011).
Keeping it. Expanding USDA index to top 3000 items to fix the rice miss.
```

---

*This guide is the fast feedback loop for Pillar 1 development. Once all 12 mini eval cases pass their eyeball test and the RAG-vs-API check is green, move on to Pillar 2.*