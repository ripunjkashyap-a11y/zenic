"""
Deterministic calculation tools. No LLM involvement — math must be exact.
The RAG explains *why* these numbers matter; these functions calculate *what* they are.
"""


_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,       # little or no exercise
    "light": 1.375,         # light exercise 1-3 days/week
    "moderate": 1.55,       # moderate exercise 3-5 days/week
    "active": 1.725,        # hard exercise 6-7 days/week
    "very_active": 1.9,     # very hard exercise + physical job
}

_MACRO_SPLITS = {
    # (protein_pct, carb_pct, fat_pct)
    "maintenance": (0.30, 0.40, 0.30),
    "cutting":     (0.40, 0.40, 0.20),
    "bulking":     (0.30, 0.50, 0.20),
}

_PROTEIN_RANGES = {
    # (min g/kg, max g/kg) based on ISSN position stands
    "sedentary":   (0.8,  1.0),
    "maintenance": (1.2,  1.6),
    "cutting":     (1.6,  2.2),
    "bulking":     (1.6,  2.2),
    "athlete":     (1.6,  2.2),
}


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor equation. Returns BMR in kcal/day."""
    if gender.lower() in ("male", "m"):
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Returns TDEE in kcal/day."""
    multiplier = _ACTIVITY_MULTIPLIERS.get(activity_level.lower())
    if multiplier is None:
        raise ValueError(f"Unknown activity_level '{activity_level}'. Choose from: {list(_ACTIVITY_MULTIPLIERS)}")
    return round(bmr * multiplier, 1)


def calculate_macros(tdee: float, goal: str) -> dict:
    """Returns grams of protein, carbs, and fat based on TDEE and goal."""
    splits = _MACRO_SPLITS.get(goal.lower())
    if splits is None:
        raise ValueError(f"Unknown goal '{goal}'. Choose from: {list(_MACRO_SPLITS)}")
    protein_pct, carb_pct, fat_pct = splits
    return {
        "protein_g": round(tdee * protein_pct / 4, 1),
        "carbs_g":   round(tdee * carb_pct   / 4, 1),
        "fat_g":     round(tdee * fat_pct     / 9, 1),
    }


def calculate_protein_range(weight_kg: float, goal: str) -> dict:
    """Returns min/max daily protein in grams based on body weight and goal (ISSN guidelines)."""
    key = goal.lower() if goal.lower() in _PROTEIN_RANGES else "maintenance"
    min_ratio, max_ratio = _PROTEIN_RANGES[key]
    return {
        "min_g": round(weight_kg * min_ratio, 1),
        "max_g": round(weight_kg * max_ratio, 1),
    }
