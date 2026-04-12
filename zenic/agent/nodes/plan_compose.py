"""Compose a structured plan (JSON) from retrieved data using the LLM."""
import json
import os
from groq import Groq
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    intent = state.get("intent")
    profile = state.get("user_profile", {})
    tool_results = state.get("tool_results", {})
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if intent == "workout_plan":
        context = "\n".join(c["text"] for c in tool_results.get("exercise_chunks", []))
        prompt = (
            f"Create a structured {tool_results.get('split_type')} workout plan for a user with goal: {profile.get('goal')}. "
            f"Available days: {profile.get('available_days')}. Equipment: {profile.get('equipment')}. "
            f"Experience: {profile.get('experience_level')}.\n\n"
            f"Available exercises:\n{context}\n\n"
            "Return a JSON object with keys: split_name, days (list of day objects with name and exercises), notes."
        )
    else:  # meal_plan
        context = "\n".join(c["text"] for c in tool_results.get("food_chunks", []))
        prompt = (
            f"Create a structured 7-day meal plan. "
            f"Macro targets: {tool_results.get('protein_min_g')}-{tool_results.get('protein_max_g')}g protein, "
            f"{tool_results.get('carbs_g')}g carbs, {tool_results.get('fat_g')}g fat, TDEE: {tool_results.get('tdee')} kcal. "
            f"Dietary restrictions: {profile.get('dietary_restrictions', 'none')}.\n\n"
            f"Available foods:\n{context}\n\n"
            "Return a JSON object with keys: daily_targets, days (list of day objects with meals), notes."
        )

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    plan_data = json.loads(response.choices[0].message.content)
    return {"plan_data": plan_data}
