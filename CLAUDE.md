# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Zenic

Zenic is a Python health and nutrition AI assistant with three pillars:

1. **Pillar 1 — Advanced RAG** (`zenic/rag/`): Answers nutrition/exercise questions from a locally indexed knowledge base using hybrid search (vector + BM25), cross-encoder re-ranking, and multi-query expansion. LLM: Groq Llama 3.3 70B.
2. **Pillar 2 — Agentic Orchestration** (`zenic/agent/`): A LangGraph `StateGraph` that routes user intent through typed nodes (safety check → router → profile check → retrieval/calculation → generate → PDF). Every node is a pure function on `ZenicState`.
3. **Pillar 3 — Evaluation** (`tests/pillar3/`, RAGAS): Faithfulness, context precision/recall, router accuracy, tool-call sequence tests, LLM-as-judge via Gemini 2.5 Flash.

UI: Streamlit (`zenic/ui/app.py`). Deployment: Streamlit Community Cloud (1 GiB RAM limit — keep models lean).

## Commands

```bash
# Install
pip install -r requirements.txt

# Run the app
streamlit run zenic/ui/app.py

# Run unit tests only (no API keys needed)
pytest tests/pillar2/test_calculations.py tests/pillar2/test_profile_check.py tests/pillar3/test_safety.py -v

# Run all tests including integration (requires GROQ_API_KEY)
pytest -v -m "not integration"          # skip LLM calls
pytest -v                               # run everything

# Pillar 1 dev scripts (require a populated vector DB)
python scripts/inspect_chunks.py
python scripts/retrieval_spot_check.py
python scripts/retrieval_spot_check.py --save baseline_v1.json
python scripts/retrieval_spot_check.py --compare baseline_v1.json
python scripts/rag_vs_api_check.py
```

Copy `.env.example` → `.env` and fill in keys before running anything that touches Groq/USDA/Qdrant.

## Architecture

### State and graph (`zenic/agent/`)

`ZenicState` (TypedDict in `state.py`) is the single shared object that flows through the LangGraph graph. Every node receives it and returns **only the fields it changes**. Key fields: `messages`, `user_profile`, `intent`, `retrieved_context`, `tool_results`, `plan_data`, `safety_flag`.

`graph.py` assembles all nodes and conditional edges. Entry point: `safety_check`. Routing logic lives in the `_route_after_*` functions at the top of that file.

Node files (`nodes/`) each export a single `run(state) -> dict` function. Nodes that need no LLM are fully deterministic (`profile_check.py`, `trend_analysis.py`).

### Retrieval pipeline (`zenic/rag/pipeline.py`)

The pipeline is: `generate_multi_queries` → `hybrid_search` (vector + BM25) → `rerank` (cross-encoder). The public entry point is `retrieve(query)`. BM25 is an in-memory index built at startup via `load_bm25_index()`.

Vector store (`vector_store.py`) switches on `ENV`: `ChromaDB` for local dev, `QdrantClient` for production. Both implement the `VectorStore` protocol.

### Calculation tools (`zenic/agent/tools/calculations.py`)

All math (BMR, TDEE, macros, protein ranges) is deterministic Python — no LLM. These are tested with exact numeric assertions. The RAG explains guidelines; these functions compute the user's specific numbers.

### Safety system

Three layers:
- **Layer 1** (`zenic/safety/layer1_classifier.py`): regex keyword filter — runs before any LLM call
- **Layer 2** (`zenic/safety/layer2_openfda.py`): live OpenFDA adverse event lookup (cached)
- **Layer 3**: system prompt constraints in `pipeline.py:generate()`

Layer 1 is always exercised for every message via the `safety_check` node.

### Ingestion (`zenic/rag/ingestion/`)

Each source has a different chunking strategy — this is intentional and important:
- **USDA**: one chunk per food item (no overlap)
- **NIH ODS**: one chunk per supplement (UL/RDA values must not be split from life-stage context)
- **ISSN papers**: section-aware chunking with a metadata prefix injected at chunk start (enables source citations)
- **wger**: one chunk per exercise (no overlap)
- **Dietary Guidelines**: recursive text splitting, 500-800 tokens, 50-100 token overlap

### Evaluation

- `eval_data/pillar1_spot_check.json`: 12 hand-crafted spot-check cases, each targeting a specific failure mode. All 12 must pass eyeball test before advancing to Pillar 2.
- Unit tests (no API key needed): `test_calculations.py`, `test_profile_check.py`, `test_safety.py`
- Integration tests (need Groq key): `test_router.py`, marked `@pytest.mark.integration`
- Pillar 3 full eval uses RAGAS (target: faithfulness > 0.85, context precision > 0.75) with Gemini as judge model.

## Key design rules

- **RAG-first, API-fallback**: always try the local vector DB first. USDA and wger live APIs are only called when local retrieval fails. The `rag_vs_api_check.py` script verifies this.
- **Deterministic tools for math, RAG for knowledge**: never let the LLM calculate TDEE, BMR, or macro splits. Those live in `calculations.py` and are tested with zero tolerance.
- **Source citations are non-negotiable**: the `generate()` function's system prompt instructs the LLM to always cite source name and year. The `p1_011` spot-check case specifically tests this.
- **Streamlit RAM budget**: the 1 GiB Streamlit Community Cloud limit is why vector storage is offloaded to Qdrant Cloud in production. Don't load large models at startup.
