"""LLM-generated insights from computed weekly stats."""
import os
from groq import Groq
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    tool_results = state.get("tool_results", {})
    stats = tool_results.get("weekly_stats", {})
    goal = (state.get("user_profile") or {}).get("goal", "general health")

    prompt = (
        f"User goal: {goal}\n"
        f"Weekly stats: {stats}\n\n"
        "Generate 3-5 specific, actionable insights from these stats. "
        "Focus on patterns, deviations from targets, and concrete recommendations. "
        "Be concise and direct — one sentence per insight."
    )
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
    )
    insights = response.choices[0].message.content
    return {
        "plan_data": {
            "weekly_stats": stats,
            "insights": insights,
        }
    }
