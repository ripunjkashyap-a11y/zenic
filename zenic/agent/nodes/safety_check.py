"""Layer 1 safety filter — runs before every LLM call."""
from zenic.agent.state import ZenicState
from zenic.safety.layer1_classifier import is_harmful


def run(state: ZenicState) -> dict:
    last_message = state["messages"][-1].content if state.get("messages") else ""
    flagged, reason = is_harmful(last_message)
    return {"safety_flag": flagged, "safety_reason": reason or ""}
