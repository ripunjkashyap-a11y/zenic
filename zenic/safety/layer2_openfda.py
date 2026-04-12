"""
Layer 2: OpenFDA adverse event lookup.
Called when a substance passes Layer 1 but the agent wants to verify safety.
Results are cached — safety status of a substance rarely changes.
"""
import os
import httpx
from functools import lru_cache

_BASE_URL = "https://api.fda.gov/drug/event.json"


@lru_cache(maxsize=256)
def check_substance(substance: str) -> dict:
    """
    Returns {"safe": bool, "adverse_event_count": int, "top_reactions": list}.
    Raises on network error — caller should handle gracefully.
    """
    params = {
        "search": f'patient.drug.medicinalproduct:"{substance}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": "5",
    }
    api_key = os.getenv("OPENFDA_API_KEY")
    if api_key:
        params["api_key"] = api_key

    response = httpx.get(_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    total_events = data.get("meta", {}).get("results", {}).get("total", 0)
    top_reactions = [r["term"] for r in results[:5]]

    # Heuristic: > 1000 adverse events is a red flag
    is_safe = total_events < 1000

    return {
        "safe": is_safe,
        "adverse_event_count": total_events,
        "top_reactions": top_reactions,
    }
