"""Determine workout split and retrieve exercises from the indexed wger data."""
from zenic.agent.state import ZenicState
from zenic.rag.pipeline import retrieve

_SPLIT_MAP = {
    3: "full_body",
    4: "upper_lower",
    5: "ulppl",
    6: "ppl",
}


def _select_split(available_days: int, goal: str) -> str:
    if goal == "fat_loss":
        return "full_body_cardio"
    return _SPLIT_MAP.get(min(available_days, 6), "full_body")


def run(state: ZenicState) -> dict:
    p = state.get("user_profile", {})
    split = _select_split(p.get("available_days", 3), p.get("goal", "maintenance"))
    equipment = p.get("equipment", "barbell")
    query = f"{split} workout exercises {equipment} {p.get('goal', '')}"
    chunks = retrieve(query)
    return {
        "tool_results": {
            **(state.get("tool_results") or {}),
            "split_type": split,
            "exercise_chunks": chunks,
        }
    }
