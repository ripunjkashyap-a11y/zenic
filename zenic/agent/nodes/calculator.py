"""Run deterministic nutrition calculations from user_profile."""
from zenic.agent.state import ZenicState
from zenic.agent.tools.calculations import (
    calculate_bmr,
    calculate_tdee,
    calculate_macros,
    calculate_protein_range,
)


def run(state: ZenicState) -> dict:
    p = state.get("user_profile", {})
    bmr = calculate_bmr(p["weight_kg"], p["height_cm"], p["age"], p["gender"])
    tdee = calculate_tdee(bmr, p["activity_level"])
    macros = calculate_macros(tdee, p["goal"])
    protein = calculate_protein_range(p["weight_kg"], p["goal"])
    return {
        "tool_results": {
            "bmr": bmr,
            "tdee": tdee,
            **macros,
            "protein_min_g": protein["min_g"],
            "protein_max_g": protein["max_g"],
        }
    }
