# Zenic V1 — Pillar 2: Agentic Orchestration (LangGraph)

> **Purpose:** This pillar transforms Zenic from a Q&A search engine into an intelligent agent that reasons, plans, and acts. LangGraph provides the control system — the LLM generates text, LangGraph governs behavior.

---

## 1. Why LangGraph?

### What LangGraph Actually Is
LangGraph is **not** a chatbot library. It's a deterministic execution engine for AI reasoning workflows. It models agent workflows as directed graphs where:
- **State** = shared memory that flows through the system
- **Nodes** = individual actions (LLM calls, tool calls, logic)
- **Edges** = control flow between nodes
- **Conditional Edges** = decision points where the agent chooses what to do next

Think of it like a flowchart that the AI navigates, with each box being a step and each arrow being a decision.

### Why Not Alternatives?
| Framework | Why Not for Zenic V1 |
|---|---|
| Plain LangChain Agents | "Loop until done" — hard to debug, no visibility into decision flow |
| CrewAI | Multi-agent (multiple personas collaborating) — overkill for a single-agent system |
| Raw Python | Works but no state management, no checkpointing, no visualization — would have to build everything from scratch |
| AutoGen | Conversation-centric, less suited for tool-heavy orchestration workflows |

**LangGraph wins because:**
- Explicit control flow — you see exactly how the agent makes decisions
- State management built in — user profile, conversation history, tool results all travel through the graph
- Visual graphs — you can generate a Mermaid diagram of your agent's logic and show it in interviews
- Production-ready — used by companies like LinkedIn, Uber, Klarna

---

## 2. Core Concepts (Learn These First)

Before designing Zenic's graph, understand these four building blocks:

### 2.1 State (The Agent's Memory)
A Python `TypedDict` that holds everything the agent needs to know at any point in execution. Every node receives the current state, does its work, and returns **only the fields it wants to update**. LangGraph merges the update into the existing state.

```python
# Conceptual example — Zenic's state
class ZenicState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    user_profile: dict          # weight, height, goals, etc.
    intent: str                 # what the user wants (classified by router)
    retrieved_context: list     # chunks from RAG
    tool_results: dict          # outputs from calculator tools
    generated_plan: dict        # structured workout/meal plan
    safety_flag: bool           # whether safety system was triggered
    current_step: str           # tracks where we are in the workflow
```

**Key principle:** Keep state minimal. Don't dump transient values into state — only persist what other nodes need to see.

### 2.2 Nodes (The Actions)
Each node is a Python function that takes state as input and returns a partial state update. Treat each node like a **pure function** — it reads from state, does one thing, and returns what changed.

```python
# Example: a simple node
def calculate_tdee(state: ZenicState) -> dict:
    profile = state["user_profile"]
    bmr = mifflin_st_jeor(profile["weight"], profile["height"], profile["age"], profile["gender"])
    tdee = bmr * activity_multipliers[profile["activity_level"]]
    return {"tool_results": {"tdee": tdee, "bmr": bmr}}
```

### 2.3 Edges (The Flow)
- **Simple edges:** "After node A, always go to node B" — `graph.add_edge("A", "B")`
- **Conditional edges:** "After node A, check the state and decide where to go" — this is where the agent's intelligence lives

```python
# Example: conditional routing
def route_after_classification(state: ZenicState) -> str:
    if state["safety_flag"]:
        return "safety_response"
    intent = state["intent"]
    if intent == "nutrition_qa":
        return "rag_retrieval"
    elif intent == "calculate":
        return "profile_check"
    elif intent == "workout_plan":
        return "profile_check"
    # ... etc
```

### 2.4 The Graph (Putting It Together)
```python
graph = StateGraph(ZenicState)
graph.add_node("router", classify_intent)
graph.add_node("rag_retrieval", retrieve_from_knowledge_base)
# ... add all nodes
graph.add_conditional_edges("router", route_after_classification, {
    "rag_retrieval": "rag_retrieval",
    "profile_check": "profile_check",
    "safety_response": "safety_response",
})
graph.set_entry_point("router")
app = graph.compile()
```

---

## 3. Zenic's Agent Architecture — The State Graph

### 3.1 High-Level Flow

```
User Message
    │
    ▼
┌─────────────────┐
│  SAFETY CHECK    │──── Triggered? ──→ Safe Response + Warning
│  (Layer 1:       │         │
│   Input Filter)  │         │
└─────────┬───────┘         │ Clean
          ▼                  │
┌─────────────────┐         │
│  ROUTER          │◄────────┘
│  (Intent         │
│   Classification)│
└─────────┬───────┘
          │
    ┌─────┼──────────┬──────────────┬──────────────┐
    ▼     ▼          ▼              ▼              ▼
  Q&A   Calculate   Meal Plan    Workout Plan   Weekly
  Flow  Flow        Flow         Flow           Summary
    │     │          │              │            Flow
    ▼     ▼          ▼              ▼              │
┌─────────────────────────────────────────┐        │
│           RESPONSE GENERATION           │◄───────┘
│  (Format final answer with citations)   │
└─────────────────────────────────────────┘
```

### 3.2 Detailed Node Definitions

Each node below is a distinct function in the codebase. Nodes are designed to do **one thing well**.

---

#### NODE: Safety Check (Entry Point)
- **What it does:** First node every message hits. Runs Layer 1 of the hard-block system — keyword filter + lightweight classification for harmful substances, steroids, dangerous dosage requests, medical diagnosis requests.
- **Input from state:** `messages` (latest user message)
- **Updates to state:** `safety_flag` (bool), `safety_reason` (str, if flagged)
- **Routes to:** If flagged → `safety_response` node. If clean → `router` node.
- **Why it's first:** Dangerous requests should never reach the LLM or any tools. This is a fast, cheap filter that catches the obvious cases before any API calls are made.

---

#### NODE: Router (Intent Classification)
- **What it does:** Uses an LLM call with structured output to classify the user's intent into one of the defined workflows. This is the "brain" that decides what the agent does.
- **Input from state:** `messages` (full conversation history for context)
- **Updates to state:** `intent` (str — one of: `nutrition_qa`, `calculate`, `meal_plan`, `workout_plan`, `weekly_summary`, `general_chat`)
- **Routes to:** Conditional edges based on `intent` value — each intent maps to a different workflow branch.
- **Implementation detail:** The LLM receives the user message + a system prompt listing the possible intents with descriptions and examples. It returns a structured JSON like `{"intent": "workout_plan", "sub_intent": "generate_split"}`. Using structured output (not free-text) makes routing deterministic and testable.

**Example classifications:**
| User Message | Intent |
|---|---|
| "How much protein is in chicken breast?" | `nutrition_qa` |
| "What's my TDEE?" | `calculate` |
| "Make me a high-protein vegetarian meal plan" | `meal_plan` |
| "Give me a ULPPL split for muscle gain" | `workout_plan` |
| "Summarize my week" | `weekly_summary` |
| "Hey, what can you do?" | `general_chat` |

---

#### NODE: Profile Check (Shared Across Workflows)
- **What it does:** Checks if `user_profile` in state has the required fields for the current workflow. Different workflows need different fields — a TDEE calculation needs weight, height, age, gender, activity level. A workout plan also needs experience level, available days, and equipment.
- **Input from state:** `user_profile`, `intent`
- **Updates to state:** `profile_complete` (bool), `missing_fields` (list)
- **Routes to:** If profile complete → next node in the workflow. If incomplete → `profile_gather` node (asks user for missing info).
- **Design decision:** This node defines required fields per intent in a config dict, not hardcoded. Makes it easy to add new workflows later.

```python
REQUIRED_FIELDS = {
    "calculate": ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal"],
    "meal_plan": ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal", "dietary_restrictions"],
    "workout_plan": ["goal", "experience_level", "available_days", "equipment"],
    "weekly_summary": []  # no profile needed, just data
}
```

---

#### NODE: Profile Gather
- **What it does:** Generates a natural-language question asking the user for missing profile information. Groups missing fields into a single friendly prompt rather than interrogating the user field-by-field.
- **Input from state:** `missing_fields`, `intent`
- **Updates to state:** `messages` (adds the assistant's question), `awaiting_input` (True)
- **Routes to:** END (the graph pauses here until the user replies in the next turn)
- **Example:** If `missing_fields = ["weight_kg", "height_cm", "age"]`, the node might produce: *"To calculate your TDEE, I need a few details — what's your weight, height, and age?"*
- **Design note:** When the user replies, the next graph invocation re-enters at the safety check → router → profile_check flow. The router should recognize the reply as providing profile data (not a new intent) and route back to profile_check, which this time sees the fields populated and proceeds.

---

#### NODE: RAG Retrieval
- **What it does:** Calls the full Pillar 1 pipeline — multi-query expansion, hybrid search, cross-encoder reranking — and returns the top 5-7 context chunks with metadata.
- **Input from state:** latest user message, optionally `user_profile` for personalization
- **Updates to state:** `retrieved_context` (list of chunks with source metadata)
- **Routes to:** `generate` node
- **Key point:** This node is a *wrapper* around Pillar 1. All the complexity of retrieval lives in Pillar 1; this node just calls it.

---

#### NODE: Generate (LLM Response Generation)
- **What it does:** Takes the user query + retrieved context + system prompt and generates the final response. Instructs the LLM to cite sources inline.
- **Input from state:** `messages`, `retrieved_context`, `user_profile`
- **Updates to state:** `messages` (appends the assistant's response)
- **Routes to:** END for Q&A workflows, or to `plan_compose`/`pdf_generate` for multi-step workflows
- **System prompt highlights:**
  - "When referencing retrieved information, always cite the source name and year"
  - "If the retrieved context doesn't contain the answer, say so — do not fabricate"
  - "Apply safety constraints from Layer 3 (no medical diagnoses, no dangerous dosages)"

---

#### NODE: Calculator Tools
- **What it does:** Calls deterministic Python functions for BMR, TDEE, macros, and protein targets. No LLM involvement in the math.
- **Input from state:** `user_profile`
- **Updates to state:** `tool_results` (dict with calculated values)
- **Routes to:** `generate` node (LLM formats the results and explains them)
- **Tools called:**
  - `calculate_bmr(weight, height, age, gender)` — Mifflin-St Jeor
  - `calculate_tdee(bmr, activity_level)` — activity multiplier
  - `calculate_macros(tdee, goal)` — percentage splits
  - `calculate_protein_range(weight, goal)` — g/kg based on goal
- **Why deterministic:** Math must be exact. A hallucinated TDEE could lead to dangerous calorie targets.

---

#### NODE: Exercise Retrieval + Split Design (Workout Plan Workflow)
- **What it does:** Determines the workout split based on `user_profile.available_days` and `user_profile.goal`, then retrieves exercises from the indexed wger data via the RAG pipeline.
- **Input from state:** `user_profile`
- **Updates to state:** `tool_results` (with `split_type`, `exercises_by_muscle_group`)
- **Routes to:** `plan_compose` node
- **Split selection logic:**
  - 3 days + any goal → Full Body
  - 4 days + hypertrophy → Upper/Lower
  - 5 days + hypertrophy → ULPPL or PPL + Upper
  - 6 days + hypertrophy → PPL (Push/Pull/Legs) twice
  - Any days + fat loss → adds cardio recommendations

---

#### NODE: Food Retrieval (Meal Plan Workflow)
- **What it does:** Retrieves foods from USDA data that match the user's macro targets and dietary restrictions. Falls back to USDA API if needed.
- **Input from state:** `user_profile.dietary_restrictions`, `tool_results.macros`
- **Updates to state:** `tool_results` (with `candidate_foods` list)
- **Routes to:** `plan_compose` node

---

#### NODE: Plan Compose
- **What it does:** Takes retrieved data (exercises or foods) and composes a structured plan using the LLM. Output is **structured JSON**, not free-form text.
- **Input from state:** `user_profile`, `tool_results`, `intent`
- **Updates to state:** `plan_data` (structured dict ready for PDF generation)
- **Routes to:** `pdf_generate` node (for workout plans) or `generate` node (for meal plans in V1)
- **Why structured JSON:** Decouples LLM reasoning from formatting. The LLM decides *what* goes in the plan; the PDF template handles *how* it looks. Also testable — you can validate the JSON against a schema.

---

#### NODE: Data Ingestion (Weekly Summary Workflow)
- **What it does:** Loads 7 days of tracking data. For V1, reads from a mock JSON file. For V2, would query a database.
- **Input from state:** date range (or defaults to "last 7 days")
- **Updates to state:** `tool_results.weekly_data` (list of daily entries)
- **Routes to:** `trend_analysis` node

---

#### NODE: Trend Analysis
- **What it does:** Runs deterministic calculations on the weekly data — averages, totals, deltas, adherence rates.
- **Input from state:** `tool_results.weekly_data`
- **Updates to state:** `tool_results.weekly_stats` (dict with computed metrics)
- **Routes to:** `insight_generation` node
- **Metrics calculated:**
  - Average daily calories and protein
  - Weekly deficit/surplus vs target
  - Weight change (first day vs last day)
  - Workout adherence rate (completed / planned)
  - Protein consistency (std deviation across days)

---

#### NODE: Insight Generation
- **What it does:** LLM analyzes the computed stats and generates actionable insights. This is the one place the LLM adds value beyond the raw numbers.
- **Input from state:** `tool_results.weekly_stats`, `user_profile.goal`
- **Updates to state:** `plan_data` (with stats + insights ready for PDF)
- **Routes to:** `pdf_generate` node
- **Example insights:**
  - "Your protein intake dropped 40% on rest days — consider a shake on non-training days"
  - "You hit your calorie target 5 of 7 days, with one significant surplus on Saturday"
  - "Workout adherence was 75% (3 of 4 planned sessions) — noting missed Friday session"

---

#### NODE: PDF Generate
- **What it does:** Takes structured `plan_data` and generates a formatted PDF using FPDF2. Returns the file path.
- **Input from state:** `plan_data`, `intent` (to pick the right template)
- **Updates to state:** `pdf_path` (string path to generated file)
- **Routes to:** `generate` node (which creates a user-facing message like "Here's your plan — I've generated a PDF you can download")
- **Templates:**
  - `workout_plan_template` — split, days, exercises with sets/reps/rest
  - `weekly_summary_template` — stats table, daily breakdown, insights section

---

#### NODE: Safe Response (Safety Intercept)
- **What it does:** When the safety check flags a harmful request, this node generates a polite, educational decline. No lecturing, no moralizing — just a brief explanation of why Zenic can't help and a suggestion to consult a professional.
- **Input from state:** `safety_reason` (what was flagged)
- **Updates to state:** `messages` (adds the safe decline response)
- **Routes to:** END
- **Example outputs:**
  - For steroid queries: *"I can't provide dosage guidance for anabolic steroids — they carry serious health risks and require medical supervision. If you're interested in maximizing natural muscle gain, I can help with training and nutrition."*
  - For extreme calorie deficits: *"Eating under 1200 kcal/day isn't something I can plan for — it's below the threshold for safe nutritional intake. If fat loss is your goal, I can help you build a sustainable moderate deficit."*

---

## 4. The Full State Schema

Here's the complete `ZenicState` TypedDict — the contract every node follows:

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class UserProfile(TypedDict, total=False):
    """User's personal data — populated progressively across the session."""
    weight_kg: float
    height_cm: float
    age: int
    gender: str                   # "male" | "female"
    activity_level: str           # "sedentary" | "light" | "moderate" | "active" | "very_active"
    goal: str                     # "muscle_gain" | "fat_loss" | "maintenance"
    dietary_restrictions: list    # ["vegetarian", "gluten_free", ...]
    experience_level: str         # "beginner" | "intermediate" | "advanced"
    available_days: int           # 2-6
    equipment: str                # "home" | "minimal" | "full_gym"

class ZenicState(TypedDict):
    """Central state — flows through every node in the graph."""

    # Conversation
    messages: Annotated[list, add_messages]

    # User data (persists across turns within a session)
    user_profile: UserProfile

    # Routing
    intent: str
    safety_flag: bool
    safety_reason: Optional[str]
    awaiting_input: bool

    # Retrieval
    retrieved_context: list

    # Tool outputs
    tool_results: dict
    plan_data: dict

    # Profile check
    profile_complete: bool
    missing_fields: list

    # Output
    pdf_path: Optional[str]
    current_step: str             # tracks where we are in the workflow (for debugging)
```

**Design principles applied:**
1. **Minimal** — only fields other nodes need to see
2. **Typed** — catches bugs at development time
3. **Reducer on messages** — `add_messages` appends rather than overwriting
4. **Separation of concerns** — conversation, profile, routing, retrieval, and output each have their own fields
5. **No dumping ground** — no generic `data: dict` field that becomes a graveyard

---

## 5. Tools Inventory — What the Agent Can Call

Tools are well-defined functions with clear input/output contracts. The agent calls them via LangGraph nodes.

### 5.1 Deterministic Calculation Tools

```
┌──────────────────────────────────────────────────────────┐
│ calculate_bmr(weight_kg, height_cm, age, gender) → float  │
│   Mifflin-St Jeor equation                                │
│   Male:   (10 × w) + (6.25 × h) − (5 × a) + 5             │
│   Female: (10 × w) + (6.25 × h) − (5 × a) − 161           │
│   Returns BMR in kcal/day                                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ calculate_tdee(bmr, activity_level) → float               │
│   sedentary × 1.2                                         │
│   light × 1.375                                           │
│   moderate × 1.55                                         │
│   active × 1.725                                          │
│   very_active × 1.9                                       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ calculate_macros(tdee, goal) → dict                       │
│   maintenance: 40% C / 30% P / 30% F                      │
│   fat_loss:    40% P / 40% C / 20% F                      │
│   muscle_gain: 40% C / 35% P / 25% F                      │
│   Returns {protein_g, carbs_g, fat_g}                     │
│   Conversions: 1g P = 4kcal, 1g C = 4kcal, 1g F = 9kcal   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ calculate_protein_range(weight_kg, goal) → dict           │
│   maintenance: 1.2-1.6 g/kg (standard active adult)       │
│   muscle_gain: 1.6-2.2 g/kg                               │
│   fat_loss:    1.6-2.4 g/kg (preserve muscle in deficit)  │
│   Returns {min_g, max_g}                                  │
│                                                           │
│   Note: 'sedentary' is an activity_level, NOT a goal.     │
│   Truly sedentary users (0.8 g/kg RDA baseline) are       │
│   handled by a separate code path keyed on                │
│   activity_level == "sedentary", not via this function.   │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Retrieval Tools

- **`rag_search(query)`** — Calls the full Pillar 1 pipeline. Returns top 5-7 reranked chunks.
- **`usda_api_lookup(food_name)`** — Fallback when RAG doesn't have the food. Called only if RAG confidence is low.
- **`wger_api_lookup(muscle_group, equipment)`** — Fallback for exercise variations not in the indexed data.

### 5.3 Safety Tools

- **`openfda_lookup(substance)`** — Queries OpenFDA adverse events and recall data. Returns `{safe: bool, warnings: list}`. Layer 2 of the hard-block system.

### 5.4 Output Tools

- **`generate_workout_pdf(plan_data)`** — Takes structured JSON, produces a formatted PDF using FPDF2. Returns file path.
- **`generate_summary_pdf(weekly_data)`** — Takes stats + insights, produces a weekly summary PDF. Returns file path.
- **`analyze_week(daily_entries)`** — Deterministic computation of weekly stats (averages, deltas, adherence).

**Design principle:** Tools are pure functions where possible. Side effects (PDF file writes) are isolated to output tools only.

---

## 6. Memory & User Profile Persistence

### 6.1 Session Memory (V1 Scope)
For V1, memory lives entirely in the LangGraph state. No database.

- **User profile** persists across turns within a session. Once a user provides their weight for a TDEE calculation, it's stored in `state.user_profile` and reused for any later request (meal plan, protein targets, weekly summary).
- **Conversation history** accumulates in `state.messages` using the `add_messages` reducer.
- **Progressive profiling** — the agent asks only for fields it needs for the current intent, never re-asks for fields already present.

### 6.2 Sliding Window Summary (Optional for V1)
If conversations get long and approach context window limits:
1. Keep the last N messages in full (e.g., 10)
2. Summarize older messages into a condensed context block
3. Prepend the summary as a system message

This is a **stretch goal** for V1 — implement only if you have time. Groq's Llama 3.3 70B has a 128K context window, so V1 conversations are unlikely to hit limits.

### 6.3 What V1 Deliberately Skips
- **Cross-session memory.** Close the browser, start fresh. V2 adds a database.
- **Long-term user preferences.** V2.
- **Conversation persistence / checkpointing.** LangGraph supports it via SqliteSaver, but V1 doesn't need it.

---

## 7. PDF Generation Architecture

### 7.1 The Key Principle: Structured JSON, Not LLM Formatting
The LLM produces a **JSON object** describing the plan. A deterministic Python template converts that JSON to a PDF. Never ask the LLM to generate formatted text directly — it's unreliable.

### 7.2 Workout Plan JSON Schema

```python
{
    "plan_name": "4-Day Upper/Lower Split",
    "goal": "muscle_gain",
    "experience_level": "intermediate",
    "duration_weeks": 4,
    "days": [
        {
            "day_number": 1,
            "name": "Upper Body A",
            "exercises": [
                {
                    "name": "Barbell Bench Press",
                    "muscle_group": "chest",
                    "sets": 4,
                    "reps": "6-8",
                    "rest_seconds": 120,
                    "notes": "Progressive overload: add 2.5kg when you hit 8 reps on all sets"
                },
                # ... more exercises
            ]
        },
        {
            "day_number": 2,
            "name": "Rest / Active Recovery",
            "exercises": [],
            "recovery_notes": "Light walking, stretching, foam rolling"
        },
        # ... more days
    ],
    "general_notes": [
        "Warm up with 5-10 min light cardio before each session",
        "Track weights each session for progressive overload"
    ]
}
```

### 7.3 Weekly Summary JSON Schema

```python
{
    "week_start": "2026-03-30",
    "week_end": "2026-04-05",
    "user_goal": "fat_loss",
    "stats": {
        "avg_daily_calories": 2228,
        "calorie_target": 2300,
        "total_deficit": -504,
        "avg_daily_protein_g": 128,
        "protein_target_g": 160,
        "workouts_completed": 4,
        "workouts_planned": 5,
        "adherence_rate": 0.80,
        "weight_start_kg": 80.2,
        "weight_end_kg": 79.8,
        "weight_change_kg": -0.4
    },
    "daily_breakdown": [ /* 7 entries */ ],
    "insights": [
        "Protein intake dropped on rest days — consider a shake on non-training days",
        "Calorie target hit on 5 of 7 days",
        "Weight trend aligns with fat loss goal (−0.4kg this week)"
    ]
}
```

### 7.4 Library Choice: FPDF2
- Pure Python, no system dependencies — works on Streamlit Community Cloud
- Simple API for tables, text, basic formatting
- Lightweight (~200KB install)
- `pip install fpdf2`

Save ReportLab for V2 if you need complex layouts or charts.

---

## 8. Mock Weekly Data for Testing

For V1 testing, create a realistic 7-day mock dataset with **intentional patterns** the agent should detect. Store as `test_data/mock_week.json`.

```json
{
    "user_profile": {
        "weight_kg": 80,
        "goal": "fat_loss",
        "calorie_target": 2300,
        "protein_target_g": 160,
        "planned_workouts": 5
    },
    "days": [
        {
            "date": "2026-03-30", "day": "Monday",
            "calories": 2250, "protein_g": 155, "carbs_g": 240, "fat_g": 70,
            "workout": {"type": "Upper Body", "completed": true, "duration_min": 65},
            "weight_kg": 80.2
        },
        {
            "date": "2026-03-31", "day": "Tuesday",
            "calories": 2100, "protein_g": 160, "carbs_g": 220, "fat_g": 68,
            "workout": {"type": "Lower Body", "completed": true, "duration_min": 60},
            "weight_kg": 80.0
        },
        {
            "date": "2026-04-01", "day": "Wednesday",
            "calories": 1900, "protein_g": 95, "carbs_g": 220, "fat_g": 60,
            "workout": {"type": "Rest Day", "completed": true, "duration_min": 0},
            "weight_kg": 80.1,
            "_pattern": "Low protein on rest day — agent should catch this"
        },
        {
            "date": "2026-04-02", "day": "Thursday",
            "calories": 2350, "protein_g": 165, "carbs_g": 250, "fat_g": 72,
            "workout": {"type": "Push", "completed": true, "duration_min": 60},
            "weight_kg": 79.9
        },
        {
            "date": "2026-04-03", "day": "Friday",
            "calories": 2200, "protein_g": 145, "carbs_g": 230, "fat_g": 68,
            "workout": {"type": "Pull", "completed": false, "duration_min": 0},
            "weight_kg": 79.8,
            "_pattern": "Missed workout — agent should note adherence drop"
        },
        {
            "date": "2026-04-04", "day": "Saturday",
            "calories": 2800, "protein_g": 130, "carbs_g": 340, "fat_g": 90,
            "workout": {"type": "Legs", "completed": true, "duration_min": 70},
            "weight_kg": 80.1,
            "_pattern": "Calorie surplus day — agent should flag this as context"
        },
        {
            "date": "2026-04-05", "day": "Sunday",
            "calories": 2000, "protein_g": 100, "carbs_g": 230, "fat_g": 65,
            "workout": {"type": "Rest Day", "completed": true, "duration_min": 0},
            "weight_kg": 79.8,
            "_pattern": "Second low-protein rest day — agent should see trend"
        }
    ]
}
```

**Expected agent insights from this data:**
1. Protein drops significantly on rest days (avg ~98g vs ~151g on training days) — recommend a shake or high-protein snack on rest days
2. Workout adherence 4/5 (80%) — note the missed Friday Pull session
3. Saturday calorie surplus (+500 kcal over target) — not alarming as one day but worth noting
4. Weekly average ~2,229 kcal vs 2,300 target → slight deficit, aligned with fat loss goal
5. Weight trend 80.2 → 79.8 (−0.4kg) → positive fat loss progress

If the LLM misses any of these patterns in its weekly summary, that's a bug worth investigating.

---

## 9. Graph Construction — The LangGraph Code

Putting it all together. This is the complete graph wiring:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ZenicState)

# Register all nodes
graph.add_node("safety_check", safety_check_node)
graph.add_node("router", router_node)
graph.add_node("safe_response", safe_response_node)
graph.add_node("profile_check", profile_check_node)
graph.add_node("profile_gather", profile_gather_node)
graph.add_node("rag_retrieval", rag_retrieval_node)
graph.add_node("calculator", calculator_node)
graph.add_node("exercise_retrieval", exercise_retrieval_node)
graph.add_node("food_retrieval", food_retrieval_node)
graph.add_node("plan_compose", plan_compose_node)
graph.add_node("data_ingestion", data_ingestion_node)
graph.add_node("trend_analysis", trend_analysis_node)
graph.add_node("insight_generation", insight_generation_node)
graph.add_node("pdf_generate", pdf_generate_node)
graph.add_node("generate", generate_node)

# Entry point
graph.add_edge(START, "safety_check")

# Safety → Router or Safe Response
graph.add_conditional_edges(
    "safety_check",
    lambda state: "unsafe" if state["safety_flag"] else "safe",
    {"unsafe": "safe_response", "safe": "router"}
)

# Router → Workflow branches
graph.add_conditional_edges(
    "router",
    lambda state: state["intent"],
    {
        "nutrition_qa": "rag_retrieval",
        "calculate": "profile_check",
        "workout_plan": "profile_check",
        "meal_plan": "profile_check",
        "weekly_summary": "data_ingestion",
        "general_chat": "generate",
    }
)

# Profile check → next step or profile gather
graph.add_conditional_edges(
    "profile_check",
    route_after_profile_check,  # reads state.profile_complete and state.intent
    {
        "gather": "profile_gather",
        "calculate": "calculator",
        "workout_plan": "exercise_retrieval",
        "meal_plan": "food_retrieval",
    }
)

# Nutrition Q&A flow
graph.add_edge("rag_retrieval", "generate")

# Calculation flow
graph.add_edge("calculator", "generate")

# Workout plan flow
graph.add_edge("exercise_retrieval", "plan_compose")
graph.add_conditional_edges(
    "plan_compose",
    lambda state: "workout" if state["intent"] == "workout_plan" else "meal",
    {"workout": "pdf_generate", "meal": "generate"}
)

# Meal plan flow (food retrieval → plan compose → generate)
graph.add_edge("food_retrieval", "plan_compose")

# Weekly summary flow
graph.add_edge("data_ingestion", "trend_analysis")
graph.add_edge("trend_analysis", "insight_generation")
graph.add_edge("insight_generation", "pdf_generate")

# PDF → final response
graph.add_edge("pdf_generate", "generate")

# Terminal nodes
graph.add_edge("generate", END)
graph.add_edge("safe_response", END)
graph.add_edge("profile_gather", END)  # Pauses graph until user replies

# Compile
app = graph.compile()
```

**To visualize the graph for interview demos:**
```python
from IPython.display import Image
Image(app.get_graph().draw_mermaid_png())
```

This generates a Mermaid diagram of the entire agent — walk a recruiter through it and they'll immediately see that you understand state machines and agent orchestration.

---

## 10. Error Handling Strategy

Tools can fail. APIs time out. The LLM returns malformed JSON. Handle it gracefully — never let an exception crash the conversation.

### 10.1 Pattern: Wrap Tool Calls
```python
def calculator_node(state: ZenicState) -> dict:
    try:
        profile = state["user_profile"]
        bmr = calculate_bmr(
            profile["weight_kg"], profile["height_cm"],
            profile["age"], profile["gender"]
        )
        tdee = calculate_tdee(bmr, profile["activity_level"])
        macros = calculate_macros(tdee, profile["goal"])
        return {
            "tool_results": {"bmr": bmr, "tdee": tdee, "macros": macros},
            "current_step": "calculation_done"
        }
    except KeyError as e:
        return {
            "tool_results": {"error": f"Missing profile field: {e}"},
            "current_step": "calculation_error"
        }
    except Exception as e:
        return {
            "tool_results": {"error": f"Calculation failed: {e}"},
            "current_step": "calculation_error"
        }
```

### 10.2 Pattern: API Retries with Fallback
```python
def usda_api_lookup(food_name: str, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            response = requests.get(USDA_URL, params={"query": food_name}, timeout=5)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.HTTPError) as e:
            if attempt == retries:
                # Graceful fallback — return None and let the agent explain
                return None
            time.sleep(1)
```

### 10.3 Pattern: Validate LLM Structured Output
```python
from pydantic import BaseModel, ValidationError

class WorkoutPlan(BaseModel):
    plan_name: str
    goal: str
    days: list

def plan_compose_node(state: ZenicState) -> dict:
    llm_output = llm.invoke(build_prompt(state))
    try:
        plan = WorkoutPlan.parse_raw(llm_output)
        return {"plan_data": plan.dict(), "current_step": "plan_composed"}
    except ValidationError as e:
        # Retry once with an error message in the prompt
        retry_output = llm.invoke(build_retry_prompt(state, str(e)))
        plan = WorkoutPlan.parse_raw(retry_output)
        return {"plan_data": plan.dict(), "current_step": "plan_composed"}
```

**Golden rule:** Every external call (LLM, API, file I/O) gets wrapped. The graph should be able to reach END on any input, even if it's an error message.

---

## 11. Tech Stack for Pillar 2

| Component | Technology | Why |
|---|---|---|
| Agent Framework | LangGraph | State machine control, visual graphs, testable nodes |
| LLM | Groq Free Tier — Llama 3.3 70B Versatile | Fast, free, good quality, tool calling support |
| PDF Generation | FPDF2 | Pure Python, lightweight, Streamlit-compatible |
| HTTP Client | `httpx` or `requests` | For USDA, wger, OpenFDA fallback calls |
| Validation | Pydantic | Structured output validation for plans |
| State Management | LangGraph TypedDict + reducers | Built-in, no extra library needed |

**Total cost: $0** — consistent with the rest of the stack.

---

## 12. Design Principles for Pillar 2

1. **The LLM reasons, tools execute, RAG provides knowledge.** Never let the LLM do math or invent facts. The router classifies intent; deterministic tools calculate; RAG retrieves grounded information.

2. **Every node is a pure function.** Read state, return partial update. No global mutation, no side effects (except output tools). This makes testing trivial.

3. **Router-first architecture.** Classify intent early, then route to specialized workflows. Don't build one giant "do everything" node.

4. **Structured output for complex deliverables.** The LLM produces JSON; templates produce PDFs. This is more reliable and testable than letting the LLM format output directly.

5. **RAG is a tool, not a pipeline step.** The agent calls the RAG pipeline when it needs knowledge. This reframes Pillar 1 as infrastructure that Pillar 2 consumes.

6. **Progressive profiling.** Ask only what you need, store it, never re-ask. This is a small detail that dramatically improves UX.

7. **Graceful degradation.** If a tool fails, the agent should still respond meaningfully. A failed API call shouldn't kill the conversation.

8. **Minimal, typed state.** No dumping-ground `data: dict`. Each field has a purpose and a type.

---

## 13. Key Interview Talking Points for This Pillar

- **"Why LangGraph over CrewAI or plain LangChain agents?"** → Zenic V1 is a single agent with multiple capabilities, not a multi-agent system. LangGraph gives explicit state-machine control. LangChain agents use a loop-until-done pattern that's hard to debug and explain. CrewAI's multi-agent abstractions would add complexity without benefit at this scale.
- **"How does the agent decide what to do?"** → A Router node classifies intent using structured LLM output. Conditional edges route to the appropriate workflow. The flow is deterministic *after* classification, which makes it both predictable and testable.
- **"Why separate calculations from the LLM?"** → LLMs are unreliable at arithmetic. A BMR calculation that's off by 200 kcal could lead to dangerous recommendations. Deterministic Python functions guarantee correctness. The LLM explains the results; tools produce them.
- **"How do you handle multi-step workflows?"** → Each step is a node. The workout plan flow has five nodes in sequence: profile check → exercise retrieval → plan compose → PDF generate → response. Data flows through state, and each node builds on the previous.
- **"How does the agent remember user info across turns?"** → Progressive profiling stored in `state.user_profile`. If the user provides their weight for a calorie calculation, it's stored and reused when they later ask for a meal plan. No re-asking.
- **"Why structured JSON for PDFs instead of letting the LLM format directly?"** → Separation of concerns. LLMs are good at deciding *what* goes in a plan; they're bad at consistent formatting. Structured JSON is schema-validatable via Pydantic. The PDF template handles formatting deterministically.
- **"What happens if a tool fails?"** → Every external call is wrapped. API failures trigger retries, then graceful fallback. LLM structured output failures trigger a retry with error context. The graph can always reach END, even on errors.

---

## 14. Risks & Mitigations for Pillar 2

| Risk | Mitigation |
|---|---|
| Router misclassifies intent | Structured output with constrained categories; test suite in Pillar 3 evaluates router accuracy; "general_chat" fallback category |
| Profile gather loop — user never provides info | Cap at 2 re-asks, then proceed with reasonable defaults or explain what's blocking |
| LLM returns invalid JSON for plans | Pydantic validation + retry-with-error-context pattern |
| External API timeouts (USDA, wger, OpenFDA) | Timeout + 1 retry + graceful fallback to cached or RAG response |
| Context window overflow on long sessions | Groq Llama 3.3 has 128K context — unlikely in V1. Sliding window summary as stretch goal |
| Agent bypasses RAG and calls APIs directly | Pillar 3 RAG Usage Rate test catches this; tune system prompt to prefer RAG |
| Workout plans are nonsensical (e.g., 5 chest exercises in one session) | Deterministic validation in plan_compose node; Pillar 3 eval flags issues |
| PDF generation fails on edge cases | Wrap in try/except; fall back to text response with "PDF generation failed" message |
| State schema drift as new features added | Strict TypedDict + Pydantic; code review any state changes |

---

## 15. Implementation Checklist

Use this as your build order for Pillar 2:

- [ ] Install LangGraph and dependencies (`pip install langgraph langchain-groq pydantic fpdf2`)
- [ ] Define the `ZenicState` TypedDict and `UserProfile` schema
- [ ] Implement deterministic calculation tools (BMR, TDEE, macros, protein range) with unit tests
- [ ] Implement the Safety Check node (Layer 1 input classifier)
- [ ] Implement the Router node with structured output for intent classification
- [ ] Implement the Profile Check + Profile Gather nodes (shared across workflows)
- [ ] Implement the RAG Retrieval node (wraps Pillar 1)
- [ ] Implement the Generate node (LLM response with source citations)
- [ ] Wire up the Nutrition Q&A workflow (router → rag → generate)
- [ ] Wire up the Calculation workflow (router → profile → calculator → generate)
- [ ] Implement the Exercise Retrieval + Plan Compose nodes for workout plans
- [ ] Implement the PDF Generate node with FPDF2 workout template
- [ ] Wire up the full Workout Plan workflow
- [ ] Implement the Food Retrieval + Meal Plan Compose nodes
- [ ] Wire up the Meal Plan workflow
- [ ] Implement the Data Ingestion + Trend Analysis + Insight Generation nodes
- [ ] Implement the weekly summary PDF template
- [ ] Wire up the full Weekly Summary workflow
- [ ] Implement the Safe Response node for the safety intercept path
- [ ] Add error handling wrappers to all tool-calling nodes
- [ ] Generate the Mermaid diagram of the final graph (for README + interviews)
- [ ] Test each workflow end-to-end with sample queries
- [ ] Hand off to Pillar 3 for formal evaluation

---

*This document covers everything needed to build Pillar 2. Pillar 1 (RAG) is consumed as a tool. Pillar 3 (Evaluation) will measure everything this pillar produces. Once Pillar 2 is implemented and passing its own testing checks, move to Pillar 3's formal evaluation pipeline.*