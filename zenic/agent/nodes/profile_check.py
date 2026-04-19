import json
import os
from groq import Groq
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
    
    if required:
        # Extract fields from the last message
        last_msg = ""
        messages = state.get("messages", [])
        if messages:
            last_msg_obj = messages[-1]
            last_msg = last_msg_obj.get("content", "") if isinstance(last_msg_obj, dict) else getattr(last_msg_obj, "content", "")
            
        if last_msg:
            prompt = (
                f"Extract physical and fitness profile data from the user's message. "
                f"Currently known profile: {json.dumps(profile)}. "
                f"Return a JSON object containing any NEW or UPDATED fields from this list: "
                f"weight_kg (number), height_cm (number), age (number), gender (string), "
                f"activity_level (MUST BE exactly one of: 'sedentary', 'light', 'moderate', 'active', 'very_active'), "
                f"goal (MUST BE exactly one of: 'maintenance', 'cutting', 'bulking'), "
                f"dietary_restrictions (string), "
                f"experience_level (string), available_days (number), equipment (string). "
                f"If none are found, return {{}}. User message: '{last_msg}'"
            )
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            try:
                extracted = json.loads(response.choices[0].message.content)
                for k, v in extracted.items():
                    if v and k in REQUIRED_FIELDS["meal_plan"] + REQUIRED_FIELDS["workout_plan"]:
                        profile[k] = v
            except:
                pass

    missing = [f for f in required if not profile.get(f)]
    return {
        "user_profile": profile,
        "profile_complete": len(missing) == 0,
        "missing_fields": missing,
    }
