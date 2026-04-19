"""Intent classification using structured LLM output."""
import json
import os

from groq import Groq
from zenic.agent.state import ZenicState

_INTENTS = ["nutrition_qa", "calculate", "meal_plan", "workout_plan", "weekly_summary", "general_chat"]
_groq_client: Groq | None = None


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

_SYSTEM_PROMPT = f"""Classify the user's message into exactly one intent.
Return a JSON object with a single key "intent" from this list: {_INTENTS}.

nutrition_qa covers ANY factual lookup answered from the knowledge base:
food nutrients, supplement guidelines, exercise descriptions, muscles worked,
equipment needed, dietary recommendations, ingredient info, etc.

Examples:
- "How much protein is in chicken?" → {{"intent": "nutrition_qa"}}
- "What muscles does the barbell row work?" → {{"intent": "nutrition_qa"}}
- "iron content in spinach" → {{"intent": "nutrition_qa"}}
- "ISSN creatine loading recommendations" → {{"intent": "nutrition_qa"}}
- "What exercises target the back with a barbell?" → {{"intent": "nutrition_qa"}}
- "vitamin D upper intake level for adults" → {{"intent": "nutrition_qa"}}
- "What's my TDEE?" → {{"intent": "calculate"}}
- "Make me a high-protein meal plan" → {{"intent": "meal_plan"}}
- "Give me a PPL workout split" → {{"intent": "workout_plan"}}
- "Summarize my week" → {{"intent": "weekly_summary"}}
- "What can you do?" → {{"intent": "general_chat"}}
- "What is Zenic?" → {{"intent": "general_chat"}}
- "Hello" → {{"intent": "general_chat"}}
- "Tell me about yourself" → {{"intent": "general_chat"}}

general_chat covers greetings, questions about Zenic itself, and anything that
is NOT a factual nutrition/exercise lookup, calculation, plan request, or summary.
"""

# LangChain message types → OpenAI/Groq API roles
_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def run(state: ZenicState) -> dict:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for msg in state.get("messages", []):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = _ROLE_MAP.get(getattr(msg, "type", "human"), "user")
            content = msg.content
        messages.append({"role": role, "content": content})

    response = _groq().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    intent = result.get("intent", "general_chat")
    return {"intent": intent}
