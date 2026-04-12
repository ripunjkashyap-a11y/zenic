"""
wger exercise database live API — fallback when exercise not found in local index.
"""
import httpx

_BASE_URL = "https://wger.de/api/v2"


def search_exercises(muscle_group: str | None = None, equipment: str | None = None, language: int = 2) -> list[dict]:
    """Returns a list of exercises. language=2 is English."""
    params: dict = {"language": language, "format": "json"}
    if muscle_group:
        params["muscles"] = muscle_group
    if equipment:
        params["equipment"] = equipment

    response = httpx.get(f"{_BASE_URL}/exercise/", params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results", [])
    return [
        {
            "name": ex.get("name"),
            "description": ex.get("description", ""),
            "muscles": ex.get("muscles", []),
            "equipment": ex.get("equipment", []),
        }
        for ex in results
    ]
