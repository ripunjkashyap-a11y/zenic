"""Check whether user_profile has all required fields for the current workflow."""
from zenic.agent.state import ZenicState

REQUIRED_FIELDS: dict[str, list[str]] = {
    "calculate":     ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal"],
    "meal_plan":     ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal", "dietary_restrictions"],
    "workout_plan":  ["goal", "experience_level", "available_days", "equipment"],
    "weekly_summary": [],
}


def run(state: ZenicState) -> dict:
    intent = state.get("intent", "")
    profile = state.get("user_profile") or {}
    required = REQUIRED_FIELDS.get(intent, [])
    missing = [f for f in required if not profile.get(f)]
    return {
        "profile_complete": len(missing) == 0,
        "missing_fields": missing,
    }
