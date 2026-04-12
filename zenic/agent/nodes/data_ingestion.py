"""Load weekly tracking data (V1: mock JSON file; V2: database query)."""
import json
import os
from datetime import date, timedelta
from zenic.agent.state import ZenicState

_MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../../data/mock_weekly.json")


def run(state: ZenicState) -> dict:
    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    try:
        with open(_MOCK_DATA_PATH) as f:
            all_entries = json.load(f)
        weekly = [
            e for e in all_entries
            if start_date.isoformat() <= e.get("date", "") <= end_date.isoformat()
        ]
    except FileNotFoundError:
        weekly = _generate_placeholder_week(start_date)
    return {"tool_results": {"weekly_data": weekly}}


def _generate_placeholder_week(start: date) -> list[dict]:
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "calories": 0,
            "protein_g": 0,
            "workout_completed": False,
        }
        for i in range(7)
    ]
