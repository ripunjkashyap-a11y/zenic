"""End-to-end TDEE pipeline test — no UI involved."""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from zenic.agent.graph import app
from zenic.agent.state import ZenicState

state = ZenicState(
    messages=[HumanMessage(content=(
        "Calculate my TDEE. I am a 25 year old male, 180cm, 80kg, "
        "and I exercise 3-5 times a week, and my goal is to maintain my current weight."
    ))],
    user_profile={},
    intent="",
    profile_complete=False,
    missing_fields=[],
    awaiting_input=False,
    retrieved_context=[],
    tool_results={},
    plan_data={},
    safety_flag=False,
    safety_reason="",
)

result = app.invoke(state)
print("=" * 60)
print(f"INTENT       : {result.get('intent')}")
print(f"PROFILE_COMP : {result.get('profile_complete')}")
print(f"USER_PROFILE : {result.get('user_profile')}")
print(f"TOOL_RESULTS : {result.get('tool_results')}")
ai_msgs = [m for m in result.get("messages", []) if getattr(m, "type", None) == "ai"]
print(f"ANSWER       :\n{ai_msgs[-1].content if ai_msgs else 'NO AI MESSAGE'}")
print("=" * 60)
