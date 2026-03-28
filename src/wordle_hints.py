"""
Structured Wordle hints from guess/feedback strings (for richer LM prompts).

Simplified letter logic (does not fully implement duplicate-letter Wordle edge cases).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def build_constraint_summary(
    previous_guesses: List[str],
    feedback_history: List[str],
) -> str:
    """
    Summarize constraints implied by all past guesses (greens / yellows / grays).

    Feedback tokens look like: "C:GREEN A:YELLOW R:GRAY ..." in order of positions.
    """
    if not previous_guesses or not feedback_history:
        return ""

    known: List[Optional[str]] = [None] * 5
    must_include: set = set()
    excluded: set = set()

    for guess, fb in zip(previous_guesses, feedback_history):
        if not fb or fb == "Invalid guess":
            continue
        guess = guess.upper().replace(" ", "")
        parts = fb.split()
        for pos, tok in enumerate(parts):
            if ":" not in tok:
                continue
            letter, color = tok.split(":", 1)
            letter = letter.strip().upper()
            color = color.strip().upper()
            if len(letter) != 1 or pos >= 5:
                continue
            if color == "GREEN":
                known[pos] = letter
            elif color == "YELLOW":
                must_include.add(letter)
            elif color == "GRAY":
                excluded.add(letter)

    for i, k in enumerate(known):
        if k is not None:
            must_include.discard(k)

    pattern = "".join((c if c else "?") for c in known)
    lines = []
    if any(c is not None for c in known):
        lines.append(f"Known pattern (position-locked greens): {pattern}")
    if must_include:
        lines.append("Must still include (yellow somewhere): " + ", ".join(sorted(must_include)))
    if excluded:
        lines.append("Letters ruled out at least once (gray): " + ", ".join(sorted(excluded)))

    if not lines:
        return ""
    return "Constraint summary:\n" + "\n".join(lines)
