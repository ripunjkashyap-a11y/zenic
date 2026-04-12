"""Return a safe refusal message when the safety flag is set."""
from zenic.agent.state import ZenicState


def run(state: ZenicState) -> dict:
    reason = state.get("safety_reason", "")
    message = (
        "I'm not able to help with that request. "
        "Zenic is designed to support healthy nutrition and fitness goals. "
        "Please consult a qualified healthcare professional for medical advice."
    )
    if reason:
        message += f" ({reason})"
    return {"messages": [{"role": "assistant", "content": message}]}
