"""Run the agent with tool-call tracing for Pillar 2 evaluation."""
import uuid
from zenic.agent.graph import app
from zenic.agent.state import ZenicState


def run_with_trace(query: str, user_profile: dict | None = None) -> dict:
    """
    Execute the graph and return the final state plus a list of nodes visited.
    Used by Pillar 2 test suite to verify tool_call sequences.

    tools_called contains the LangGraph node names in visit order, plus any
    virtual tool names injected from state (e.g. "usda_api" when the RAG
    fallback triggers) so that rag_vs_api_check.py can detect them.
    """
    trace_id = str(uuid.uuid4())
    initial_state: ZenicState = {
        "messages": [{"role": "user", "content": query}],
        "user_profile": user_profile or {},
        "intent": "",
        "profile_complete": False,
        "missing_fields": [],
        "awaiting_input": False,
        "retrieved_context": [],
        "tool_results": {},
        "plan_data": {},
        "safety_flag": False,
        "safety_reason": "",
    }

    nodes_visited = []
    accumulated_tool_results: dict = {}
    final_state = None

    for step in app.stream(initial_state):
        node_name = list(step.keys())[0]
        nodes_visited.append(node_name)
        partial = step[node_name]
        # Accumulate tool_results across all nodes so nothing is lost when a
        # later node (e.g. generate) doesn't re-emit tool_results.
        if isinstance(partial.get("tool_results"), dict):
            accumulated_tool_results.update(partial["tool_results"])
        final_state = partial

    # Expose API fallback as a virtual tool name so callers can detect it with
    # a simple `"usda_api" in tools_called` check (mirrors node-name convention).
    api_fallback = accumulated_tool_results.get("api_fallback_used")
    if api_fallback:
        nodes_visited.append(api_fallback)

    if final_state is not None:
        final_state["tool_results"] = accumulated_tool_results

    return {
        "id": trace_id,
        "tools_called": nodes_visited,
        "final_state": final_state,
    }
