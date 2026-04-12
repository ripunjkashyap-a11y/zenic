"""
Layer 1: Keyword filter + lightweight classification.
Fast, cheap — catches obvious harmful requests before any LLM or API calls.
"""
import re

_HARD_BLOCK_PATTERNS = [
    r"\bsteroids?\b",
    r"\bsarms?\b",
    r"\bgrowth hormone\b",
    r"\bhgh\b",
    r"\bephedrine\b",
    r"\bdnp\b",
    r"\bdinitrophenol\b",
    r"\bclenbuterol\b",
    r"\bmedical diagnos\w+\b",
    r"\bcure\s+(my\s+)?(cancer|diabetes|heart)",
    r"\b(500|400|300)\s*calories?\s*(a\s+day|per\s+day|daily)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _HARD_BLOCK_PATTERNS]


def is_harmful(text: str) -> tuple[bool, str | None]:
    """Returns (is_harmful, reason_or_None)."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return True, f"Matched restricted term: {pattern.pattern}"
    return False, None
