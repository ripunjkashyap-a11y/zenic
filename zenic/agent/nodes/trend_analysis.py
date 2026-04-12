"""Deterministic weekly stats from raw tracking data."""
import statistics
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    tool_results = state.get("tool_results", {})
    data = tool_results.get("weekly_data", [])

    if not data:
        return {"tool_results": {**tool_results, "weekly_stats": {}}}

    calories = [d.get("calories", 0) for d in data]
    protein = [d.get("protein_g", 0) for d in data]
    workouts_planned = sum(1 for d in data if d.get("workout_planned", False))
    workouts_done = sum(1 for d in data if d.get("workout_completed", False))

    stats = {
        "avg_calories": round(statistics.mean(calories), 1),
        "avg_protein_g": round(statistics.mean(protein), 1),
        "protein_consistency_std": round(statistics.stdev(protein), 1) if len(protein) > 1 else 0,
        "workout_adherence_pct": round(workouts_done / workouts_planned * 100) if workouts_planned else None,
        "weight_change_kg": (
            round(data[-1].get("weight_kg", 0) - data[0].get("weight_kg", 0), 2)
            if data[0].get("weight_kg") and data[-1].get("weight_kg") else None
        ),
    }
    return {"tool_results": {**tool_results, "weekly_stats": stats}}
