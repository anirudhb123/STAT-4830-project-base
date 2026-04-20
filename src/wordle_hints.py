"""
Structured Wordle hints from guess/feedback strings (for richer LM prompts).

Simplified letter logic (does not fully implement duplicate-letter Wordle edge cases).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


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


def _parse_constraints(
    previous_guesses: Sequence[str],
    feedback_history: Sequence[str],
) -> Tuple[List[Optional[str]], set, List[set], set]:
    """Internal: turn (guesses, feedback) into structured constraint sets.

    Returns ``(known, must_include, must_not_at, excluded)`` where:

    - ``known[i]`` is the green-locked letter at position i, or None.
    - ``must_include`` is the set of letters that must appear somewhere (yellow).
    - ``must_not_at[i]`` is the set of letters that cannot appear at position i
      (yellow at position i; also any green's letter cannot appear there from a
      different position). Non-position-specific yellow info is captured in
      ``must_include``.
    - ``excluded`` is the set of letters fully ruled out (gray, with greens and
      yellows removed so duplicate-letter cases don't accidentally exclude a
      letter that's known to appear).
    """
    known: List[Optional[str]] = [None] * 5
    must_include: set = set()
    must_not_at: List[set] = [set() for _ in range(5)]
    excluded: set = set()

    for guess, fb in zip(previous_guesses, feedback_history):
        if not fb or fb == "Invalid guess":
            continue
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
                must_not_at[pos].add(letter)
            elif color == "GRAY":
                excluded.add(letter)

    # A letter that's also locked in greens / yellows shouldn't be globally excluded
    # (this is the simplified-duplicate-letter handling -- good enough for our
    # warm-start signal even though it isn't a perfect Wordle solver).
    excluded -= {c for c in known if c is not None}
    excluded -= must_include
    return known, must_include, must_not_at, excluded


def find_consistent_words(
    candidates: Iterable[str],
    previous_guesses: Sequence[str],
    feedback_history: Sequence[str],
) -> List[str]:
    """Filter ``candidates`` to those consistent with all (guess, feedback) constraints.

    Uses the same simplified Wordle constraints as ``build_constraint_summary``
    (no full duplicate-letter edge case handling -- a letter ruled gray is only
    excluded outright if it has no green/yellow elsewhere). Returns the subset
    of ``candidates`` (uppercased) that satisfies every constraint, preserving
    the original iteration order.

    With no prior guesses/feedback, returns ``list(candidates)`` (uppercased,
    5-letter alphabetic only) unchanged.
    """
    cleaned = [w.upper() for w in candidates if isinstance(w, str) and len(w) == 5 and w.isalpha()]
    if not previous_guesses or not feedback_history:
        return cleaned

    known, must_include, must_not_at, excluded = _parse_constraints(
        previous_guesses, feedback_history
    )

    out: List[str] = []
    for u in cleaned:
        ok = True
        for i, k in enumerate(known):
            if k is not None and u[i] != k:
                ok = False
                break
        if not ok:
            continue
        if any(letter not in u for letter in must_include):
            continue
        if any(letter in u for letter in excluded):
            continue
        if any(u[i] in banned for i, banned in enumerate(must_not_at)):
            continue
        out.append(u)
    return out
