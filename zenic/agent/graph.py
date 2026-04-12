"""
LangGraph StateGraph for Zenic.

Entry point: safety_check
Workflows:
  nutrition_qa     → rag_retrieval → generate → END
  calculate        → profile_check → [profile_gather → END | calculator → generate → END]
  meal_plan        → profile_check → [profile_gather → END | food_retrieval → plan_compose → generate → END]
  workout_plan     → profile_check → [profile_gather → END | exercise_retrieval → plan_compose → pdf_generate → END]
  weekly_summary   → data_ingestion → trend_analysis → insight_generation → pdf_generate → END
  general_chat     → generate → END
"""
from langgraph.graph import StateGraph, END

from zenic.agent.state import ZenicState
from zenic.agent.nodes import (
    safety_check,
    router,
    profile_check,
    profile_gather,
    rag_retrieval,
    calculator,
    exercise_retrieval,
    food_retrieval,
    plan_compose,
    pdf_generate,
    data_ingestion,
    trend_analysis,
    insight_generation,
    generate,
    safety_response,
)


def _route_after_safety(state: ZenicState) -> str:
    return "safety_response" if state.get("safety_flag") else "router"


def _route_after_router(state: ZenicState) -> str:
    intent = state.get("intent", "general_chat")
    if intent == "nutrition_qa":
        return "rag_retrieval"
    if intent in ("calculate", "meal_plan", "workout_plan"):
        return "profile_check"
    if intent == "weekly_summary":
        return "data_ingestion"
    return "generate"  # general_chat


def _route_after_profile_check(state: ZenicState) -> str:
    if not state.get("profile_complete"):
        return "profile_gather"
    intent = state.get("intent")
    if intent == "calculate":
        return "calculator"
    if intent == "meal_plan":
        return "food_retrieval"
    if intent == "workout_plan":
        return "exercise_retrieval"
    return "generate"


def build_graph() -> StateGraph:
    g = StateGraph(ZenicState)

    g.add_node("safety_check",      safety_check.run)
    g.add_node("router",            router.run)
    g.add_node("profile_check",     profile_check.run)
    g.add_node("profile_gather",    profile_gather.run)
    g.add_node("rag_retrieval",     rag_retrieval.run)
    g.add_node("calculator",        calculator.run)
    g.add_node("exercise_retrieval", exercise_retrieval.run)
    g.add_node("food_retrieval",    food_retrieval.run)
    g.add_node("plan_compose",      plan_compose.run)
    g.add_node("pdf_generate",      pdf_generate.run)
    g.add_node("data_ingestion",    data_ingestion.run)
    g.add_node("trend_analysis",    trend_analysis.run)
    g.add_node("insight_generation", insight_generation.run)
    g.add_node("generate",          generate.run)
    g.add_node("safety_response",   safety_response.run)

    g.set_entry_point("safety_check")

    g.add_conditional_edges("safety_check", _route_after_safety, {
        "safety_response": "safety_response",
        "router":          "router",
    })
    g.add_conditional_edges("router", _route_after_router, {
        "rag_retrieval":  "rag_retrieval",
        "profile_check":  "profile_check",
        "data_ingestion": "data_ingestion",
        "generate":       "generate",
    })
    g.add_conditional_edges("profile_check", _route_after_profile_check, {
        "profile_gather":    "profile_gather",
        "calculator":        "calculator",
        "food_retrieval":    "food_retrieval",
        "exercise_retrieval": "exercise_retrieval",
        "generate":          "generate",
    })

    g.add_edge("rag_retrieval",      "generate")
    g.add_edge("calculator",         "generate")
    g.add_edge("food_retrieval",     "plan_compose")
    g.add_edge("exercise_retrieval", "plan_compose")
    g.add_edge("plan_compose",       "pdf_generate")
    g.add_edge("data_ingestion",     "trend_analysis")
    g.add_edge("trend_analysis",     "insight_generation")
    g.add_edge("insight_generation", "pdf_generate")
    g.add_edge("generate",           END)
    g.add_edge("pdf_generate",       END)
    g.add_edge("profile_gather",     END)
    g.add_edge("safety_response",    END)

    return g


app = build_graph().compile()
