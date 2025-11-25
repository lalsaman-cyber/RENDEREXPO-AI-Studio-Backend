# app/core/safety.py
"""
Very simple prompt safety checker for RENDEREXPO AI STUDIO.

IMPORTANT:
- This is a *starter* safety layer.
- Later we can replace/extend it with a real policy engine.
- For now, it just blocks obviously bad / disallowed content.

This helps you:
- Stay closer to Stability AI AUP + OpenRAIL-style restrictions.
- Prove you have a content filter in place.
"""

from typing import Tuple, Optional, List


# VERY SIMPLE keyword blocklist.
# We keep it high-level and obvious (no graphic details).
DISALLOWED_KEYWORDS: List[str] = [
    "child porn",
    "cp",
    "underage",
    "under-age",
    "kidnapping",
    "terrorist",
    "terrorism",
    "bomb making",
    "make a bomb",
    "self harm",
    "suicide",
    "kill myself",
    "kill him",
    "kill her",
    "beheading",
    "graphic gore",
    "disemboweled",
    "neo-nazi",
    "white supremacist",
    "hate symbol",
]


def check_prompt_safety(
    prompt: str,
    negative_prompt: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Basic safety check.

    Returns:
        (is_safe, reason_if_not_safe)

    Logic:
    - Lowercases the combined prompt + negative_prompt
    - Checks if any DISALLOWED_KEYWORDS appear
    - If found → unsafe
    """
    combined = f"{prompt or ''} {negative_prompt or ''}".lower()

    found = [kw for kw in DISALLOWED_KEYWORDS if kw in combined]
    if found:
        reason = (
            "Prompt contains disallowed content keywords: "
            + ", ".join(sorted(set(found)))
        )
        return False, reason

    # Later we can add:
    # - length checks
    # - regex checks
    # - category-based checks
    return True, None
