"""Ask the user for missing profile fields in a single friendly message."""
import os
from groq import Groq
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    missing = state.get("missing_fields", [])
    intent = state.get("intent", "")
    prompt = (
        f"The user wants a {intent.replace('_', ' ')}. "
        f"You need the following information to proceed: {', '.join(missing)}. "
        "Ask the user for all of these in a single, natural, friendly message."
    )
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
    )
    question = response.choices[0].message.content
    return {
        "messages": [{"role": "assistant", "content": question}],
        "awaiting_input": True,
    }
