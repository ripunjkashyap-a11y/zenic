"""
USDA FoodData Central live API — fallback when item not found in local index.
Only called when RAG retrieval returns no relevant results for a food query.
"""
import os
import httpx

_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def search_food(query: str, page_size: int = 5) -> list[dict]:
    """Returns a list of foods with nutrient summaries."""
    response = httpx.get(
        f"{_BASE_URL}/foods/search",
        params={"query": query, "pageSize": page_size, "api_key": os.environ["USDA_API_KEY"]},
        timeout=10,
    )
    response.raise_for_status()
    items = response.json().get("foods", [])
    return [
        {
            "name": item.get("description"),
            "fdcId": item.get("fdcId"),
            "nutrients": {
                n["nutrientName"]: f"{n['value']} {n['unitName']}"
                for n in item.get("foodNutrients", [])[:10]
            },
        }
        for item in items
    ]
