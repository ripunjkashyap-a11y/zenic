import os, json, httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

key = os.environ["USDA_API_KEY"]
print("API key found:", bool(key))

# Test /foods/list
r = httpx.post(
    "https://api.nal.usda.gov/fdc/v1/foods/list",
    params={"api_key": key},
    json={"dataType": ["Foundation"], "pageSize": 3, "pageNumber": 1},
)
print("POST /foods/list status:", r.status_code)
data = r.json()
print("Response type:", type(data).__name__)
if isinstance(data, list):
    print("Items:", len(data))
    if data:
        print(json.dumps(data[0], indent=2)[:600])
else:
    print(json.dumps(data, indent=2)[:600])

# Also try GET /foods/search to check a specific food
print("\n--- Testing /foods/search ---")
r2 = httpx.get(
    "https://api.nal.usda.gov/fdc/v1/foods/search",
    params={"api_key": key, "query": "chicken breast", "pageSize": 1},
)
print("GET /foods/search status:", r2.status_code)
s = r2.json()
foods = s.get("foods", [])
print("Foods returned:", len(foods))
if foods:
    print("First food keys:", list(foods[0].keys()))
    print("Nutrients count:", len(foods[0].get("foodNutrients", [])))
