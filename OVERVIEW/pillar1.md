# Zenic V1 — Pillar 1: Nutritional Guide (Advanced RAG)

> **Purpose:** This is the technical foundation of Zenic. A Retrieval-Augmented Generation system that answers health, nutrition, and exercise questions with grounded, cited, and safe responses.

---

## 1. Knowledge Base — Data Sources

All sources are **free and open access**. No paid APIs or subscriptions required.

### 1.1 Structured Nutritional Data — USDA FoodData Central
- **Source:** [https://fdc.nal.usda.gov/](https://fdc.nal.usda.gov/)
- **Access:** Free API key (register on the site)
- **Strategy:** Download bulk CSV/JSON data dumps. Do NOT rely on live API at query time.
- **Contents:** Thousands of foods with full macro/micronutrient breakdowns (calories, protein, carbs, fat, vitamins, minerals per serving)
- **RAG Role:** Primary retrieval source for food-related queries
- **API Fallback:** The agent can call the USDA API as a **tool** when a food item isn't found in the local index. This must be tested to ensure the agent doesn't over-rely on API calls — if it does, there's no point in the RAG.

### 1.2 Nutritional Limits, RDAs & Supplement Boundaries — NIH Office of Dietary Supplements
- **Source:** [https://ods.od.nih.gov/](https://ods.od.nih.gov/)
- **Access:** Free, public domain
- **Contents:** Comprehensive fact sheets with:
  - Upper Intake Levels (ULs) for vitamins and minerals
  - Recommended Dietary Allowances (RDAs) broken down by age, gender, and condition (pregnancy, lactation)
  - Supplement safety information
- **Important Note:** Some values are **body-weight dependent** (e.g., protein: 0.8g/kg sedentary → 1.6-2.2g/kg for muscle building). These dynamic calculations should be handled by **deterministic agent tools**, not retrieved text. The RAG provides the guidelines; the tool does the math with the user's actual weight.

### 1.3 Drug & Supplement Safety — OpenFDA
- **Source:** [https://open.fda.gov/](https://open.fda.gov/)
- **Access:** Free. No API key required (rate-limited to 240 req/min without key, 120,000/day with free key)
- **Contents:** Adverse event reports, recall data, drug interactions
- **RAG Role:** Not indexed into the vector DB. Used as a **live tool** the agent calls when a user mentions a substance. Feeds into the 3-layer hard-block system (see Section 5).

### 1.4 Dietary Guidelines
- **Sources:**
  - Dietary Guidelines for Americans — [https://www.dietaryguidelines.gov/](https://www.dietaryguidelines.gov/) (Free, public domain)
  - WHO Dietary Guidelines (Free, public domain)
- **Contents:** Guidelines for different goals (cutting, bulking, maintenance), different diets (keto, Mediterranean, vegetarian, vegan), gender-specific recommendations based on height, weight, and activity level
- **RAG Role:** Long-form text documents indexed into the vector DB with section-aware chunking
- **Calculation Note:** TDEE, BMR (Mifflin-St Jeor equation), macro splits (40/30/30 maintenance, 40/40/20 cutting, etc.) are **deterministic formulas** — these live as agent tools, not as RAG content. The RAG explains *why* these matter; the tools calculate *your specific numbers*.

### 1.5 Exercise Database — wger
- **Source:** [https://wger.de/](https://wger.de/) (Open source exercise database with REST API)
- **Access:** Free, open source
- **Strategy:** Download and index exercise data into the vector DB for fast retrieval when building workout plans. Keep the live wger API as a fallback tool for edge cases or exercise variations.
- **Contents:** Exercises categorized by muscle group, equipment needed, and difficulty

### 1.6 Exercise Nutrition — ISSN Position Papers
- **Source:** PubMed Central — International Society of Sports Nutrition (ISSN) position stands
- **Access:** Freely available (open-access peer-reviewed papers)
- **Contents:** Pre/post workout nutrition, protein timing, supplement basics (creatine, caffeine), hydration guidelines, how macros interact with training goals
- **Chunking Strategy:** These are **long-form scientific documents** — they require a different chunking approach than food data (see Section 3).
- **Metadata:** Each chunk must include: source title, authors, publication year, and section name — this enables source citations in generated responses.

---

## 2. Retrieval Pipeline — Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Processing    │
│  (Multi-Query or     │
│   HyDE Expansion)    │
└─────────┬───────────┘
          │ generates 3-4 query variants
          ▼
┌─────────────────────┐
│   Hybrid Search      │
│  ┌───────┐ ┌──────┐ │
│  │Vector │ │ BM25 │ │
│  │Search │ │Search│ │
│  └───┬───┘ └──┬───┘ │
│      └────┬───┘     │
│     Merge Results    │
└─────────┬───────────┘
          │ ~15-20 candidates
          ▼
┌─────────────────────┐
│   Cross-Encoder      │
│   Re-ranking         │
│   (BGE-reranker or   │
│    ms-marco-MiniLM)  │
└─────────┬───────────┘
          │ top 5-7 results
          ▼
┌─────────────────────┐
│   LLM Generation     │
│   (Groq - Llama 3.3  │
│    70B Versatile)     │
│   + Source Citations  │
└─────────────────────┘
```

---

## 3. Retrieval Pipeline — Step-by-Step Detail

### Step 1: Ingestion & Chunking

Different source types require different chunking strategies. This is a key design decision and a strong interview talking point.

| Source Type | Chunking Strategy | Chunk Size | Overlap | Metadata Tags |
|---|---|---|---|---|
| USDA Food Data | One document per food item with full nutritional profile | Varies (~200-400 tokens) | None | category, food_group, source="USDA" |
| Dietary Guidelines | Recursive text splitting | 500-800 tokens | 50-100 tokens | category, topic, source, section |
| ISSN Papers | **Section-aware chunking** with context headers | 800-1000 tokens | 100 tokens | source_title, authors, year, section, topic |
| wger Exercises | One document per exercise | Varies (~150-300 tokens) | None | muscle_group, equipment, difficulty, source="wger" |
| NIH Supplement Data | One document per supplement/nutrient | Varies (~300-600 tokens) | None | nutrient_name, category, source="NIH_ODS" |

**Why section-aware chunking for ISSN papers?** Scientific text loses meaning when fragmented too small. Each chunk gets a metadata prefix like:
> "Source: ISSN Position Stand on Protein and Exercise (Jäger et al., 2017), Section: Recommendations"

This enables accurate source citations in the final response.

### Step 2: Dual Indexing for Hybrid Search

Every chunk gets indexed **two ways**:

**Vector Index (Semantic Search):**
- Embedding model: `all-MiniLM-L6-v2` (fast, ~80MB) or `BGE-small-en-v1.5` (better quality, still small)
- Both run on CPU, no GPU needed, completely free
- Stored in: Qdrant Cloud (production) / ChromaDB (local dev)

**BM25 Keyword Index (Lexical Search):**
- Library: `rank_bm25` Python package
- Purpose: Catches exact term matches that semantic search misses

**Why hybrid?** Semantic search alone fails on specific terms. Example:
- Query: "how much iron is in spinach?"
- Vector search might return generically similar nutrition content
- BM25 nails "iron" + "spinach" exactly
- Hybrid gives the best of both

### Step 3: Query Processing (Biggest Differentiator)

Before hitting the database, transform the user's query to improve retrieval.

**Recommended approach: Multi-Query Generation**
- Take the original question and generate 3-4 alternative phrasings using the LLM
- Example: "high protein vegan foods" becomes:
  - "plant-based protein sources"
  - "vegan foods with most protein per serving"
  - "protein-rich foods without animal products"
- Retrieve for ALL variants and merge results
- **Why multi-query over HyDE?** More intuitive to explain in interviews, results are easier to evaluate, and it's more transparent

**Alternative considered: HyDE (Hypothetical Document Embeddings)**
- Ask the LLM to generate a hypothetical answer, embed *that*, and use it for retrieval
- Intuition: the hypothetical answer is semantically closer to actual answer documents than the question
- Can mention this as an alternative approach you considered — shows awareness of the landscape

### Step 4: Cross-Encoder Re-ranking

- Initial hybrid search returns ~15-20 candidates
- Pass through a cross-encoder model that scores each candidate against the original query
- Models: `BGE-reranker-base` or `ms-marco-MiniLM-L-6-v2` (~80MB, runs on CPU)
- Keep top 5-7 after re-ranking
- **This is visually demonstrable** — you can show "here's what retrieval returned vs. what re-ranking kept" in your eval pipeline

### Step 5: Generation with Source Citations

- LLM (Groq — Llama 3.3 70B) receives: re-ranked context chunks + user query + system prompt
- System prompt instructs: "When referencing retrieved information, always state the source name and year."
- Example output style:
  > "According to the ISSN Position Stand on Protein and Exercise (Jäger et al., 2017), athletes engaged in resistance training benefit from 1.6-2.2g of protein per kilogram of body weight."
- Source attribution is not just a UI feature — it's directly tied to **faithfulness evaluation** in Pillar 3

---

## 4. Vector Database Strategy

### Production (Streamlit Community Cloud Deployment)
- **Qdrant Cloud — Free Forever Tier** (1GB storage)
- Reason: Streamlit Community Cloud has a **1 GiB RAM limit**. ChromaDB in-process + sentence-transformers would risk exceeding it. Qdrant Cloud keeps the app lightweight — just API calls.
- Supports hybrid search natively, great filtering, open source

### Local Development
- **ChromaDB — In-Process**
- `pip install chromadb` — no Docker, no separate server
- Runs embedded in the Python application
- Good for rapid iteration and testing

### Environment Switching
```python
# Example pattern
if ENV == "production":
    vector_store = QdrantCloudClient(url=QDRANT_URL, api_key=QDRANT_KEY)
elif ENV == "development":
    vector_store = ChromaDBClient(persist_directory="./chroma_db")
```
This pattern (swap vector DB based on environment) is worth mentioning in interviews — it shows real-world engineering thinking.

---

## 5. Hard-Block System for Harmful Substances (3-Layer Filter)

Health is a sensitive domain. The agent must **never** recommend harmful drugs, dangerous dosages, or provide medical diagnoses.

### Layer 1: Input Classifier
- Catches obvious requests for harmful substances, steroids, dangerous dosages **before** they reach the agent
- Can be a simple keyword filter + lightweight classifier
- Fast, low-cost first line of defense

### Layer 2: OpenFDA Lookup
- If a substance gets through Layer 1, the agent queries OpenFDA for adverse events and safety data
- Blocks response if the substance is flagged for serious adverse events
- This is a **real safety check**, not just a keyword list — much more impressive in interviews

### Layer 3: System Prompt Constraints
- Final safety net with explicit instructions:
  - Never recommend controlled substances or steroids
  - Never provide medical diagnoses
  - Flag dangerously low calorie targets (under 1200 kcal)
  - Always recommend consulting a healthcare professional for medical questions
  - Never recommend supplement dosages above established Upper Intake Levels (ULs)

**Interview talking point:** This 3-layer approach shows defense-in-depth thinking. Each layer catches what the previous one might miss.

---

## 6. Tech Stack Summary (All Free)

| Component | Technology | Cost |
|---|---|---|
| LLM (Generation + Agent) | Groq Free Tier — Llama 3.3 70B Versatile | Free |
| LLM (Evaluation / Judge) | Google Gemini Free Tier — 2.5 Flash | Free |
| Vector DB (Production) | Qdrant Cloud — Free Tier | Free |
| Vector DB (Local Dev) | ChromaDB — In-Process | Free |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2` or `BGE-small-en-v1.5`) | Free (local) |
| BM25 Keyword Search | `rank_bm25` Python package | Free |
| Cross-Encoder Reranker | `BGE-reranker-base` or `ms-marco-MiniLM-L-6-v2` | Free (local) |
| Nutritional Data | USDA FoodData Central (bulk download + API fallback) | Free |
| Supplement Safety | NIH Office of Dietary Supplements | Free |
| Drug Safety | OpenFDA API | Free |
| Dietary Guidelines | DietaryGuidelines.gov + WHO | Free |
| Exercise Database | wger API + indexed data | Free |
| Exercise Nutrition | ISSN Position Papers (PubMed Central) | Free |
| Deployment | Streamlit Community Cloud | Free |

**Total cost: $0**

---

## 7. Design Principles to Remember

1. **RAG-first, API-fallback pattern:** Always try the local vector DB first. Only call external APIs (USDA, wger) when the local index doesn't have the answer. Test this — if the agent over-relies on API calls, the RAG is pointless.

2. **Deterministic tools for math, RAG for knowledge:** TDEE calculations, macro splits, protein-per-kg math — these are Python functions, not LLM tasks. The RAG explains *why*; the tools calculate *what*.

3. **Heterogeneous chunking:** Different data types need different strategies. Food data ≠ scientific papers ≠ exercise descriptions. This is a design decision you should be able to articulate clearly.

4. **Source attribution is non-negotiable:** Every response referencing retrieved content must cite the source. This builds user trust and feeds directly into faithfulness evaluation (Pillar 3).

5. **Safety is a feature, not an afterthought:** The 3-layer hard-block system shows engineering maturity. Health misinformation can cause real harm.

6. **Document the data pipeline:** How you collected, cleaned, chunked, and indexed the knowledge base is interview gold. Most candidates gloss over this, but it's where real-world RAG projects spend 60% of their time.

---

## 8. Key Interview Talking Points for This Pillar

- **"Why hybrid search?"** → Semantic search alone fails on specific terms (e.g., "iron in spinach"). BM25 catches exact matches. The combination gives precision + recall.
- **"Why multi-query over HyDE?"** → More transparent, easier to evaluate, and results are more interpretable. HyDE is a valid alternative but adds a layer of abstraction.
- **"Why re-ranking?"** → Initial retrieval is recall-optimized (get lots of candidates). Re-ranking is precision-optimized (keep only the best). Cross-encoders are more accurate than bi-encoders because they see the query and document together.
- **"Why Qdrant Cloud over ChromaDB in production?"** → Streamlit's 1GB RAM limit. Offloading vector storage to a cloud service keeps the app lightweight. ChromaDB is used locally for fast development iteration.
- **"How do you handle body-weight-dependent values?"** → Deterministic tools, not RAG. The agent retrieves the guideline ("1.6-2.2g/kg for muscle building") and then calls a calculation tool with the user's actual weight to produce a personalized number.
- **"How do you ensure the agent doesn't just call the API instead of using RAG?"** → Testing and evaluation. Part of the eval pipeline checks retrieval hit rates. If the agent bypasses RAG too often, it indicates indexing gaps that need to be fixed.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Groq free tier rate limits hit during demo | Implement semantic caching for repeated/similar queries |
| USDA bulk data is too large to fully index | Index the most common 2000-3000 foods; API fallback handles the rest |
| Streamlit 1GB RAM limit | Qdrant Cloud for vectors; only load small embedding + reranker models |
| Scientific papers contain outdated info | Tag chunks with publication year; instruct LLM to prefer recent sources |
| Agent hallucinates nutritional values | Faithfulness evaluation in Pillar 3 catches this; source citations make it verifiable |
| OpenFDA rate limits during safety checks | Cache safety check results; most substances won't change safety status frequently |

---

*This document covers everything needed to build Pillar 1. Proceed to Pillar 2 (Agentic Orchestration with LangGraph) once implementation planning is complete.*