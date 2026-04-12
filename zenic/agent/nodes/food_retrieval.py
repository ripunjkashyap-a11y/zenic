"""Retrieve foods matching macro targets and dietary restrictions."""
from zenic.agent.state import ZenicState
from zenic.rag.pipeline import retrieve


def run(state: ZenicState) -> dict:
    p = state.get("user_profile", {})
    tool_results = state.get("tool_results") or {}
    restrictions = p.get("dietary_restrictions", "none")
    goal = p.get("goal", "maintenance")
    protein_target = tool_results.get("protein_min_g", "")
    query = f"high protein foods {restrictions} diet {goal} {protein_target}g protein"
    chunks = retrieve(query)
    return {
        "tool_results": {
            **tool_results,
            "food_chunks": chunks,
        }
    }
