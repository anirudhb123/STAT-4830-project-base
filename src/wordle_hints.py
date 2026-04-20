"""
Structured Wordle hints from guess/feedback strings (for richer LM prompts).

Handles Wordle's duplicate-letter semantics: greens consume target slots
first, yellows consume from what remains, any extra copies of the same
letter in the guess come back GRAY. This means a letter marked GRAY in
one position of a guess can still appear in the target — just not more
times than the green+yellow count of that letter *in that same guess*.

The exported parser tracks per-letter ``min_count`` / ``max_count`` bounds
so both ``build_constraint_summary`` (LM prompt text) and
``find_consistent_words`` (supervised warm-start consistency filter)
agree on the correct semantics at vocab sizes where duplicate letters
actually show up.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _parse_constraints(
    previous_guesses: Sequence[str],
    feedback_history: Sequence[str],
) -> Tuple[
    List[Optional[str]],
    Set[str],
    List[Set[str]],
    Set[str],
    Dict[str, int],
    Dict[str, int],
]:
    """Turn (guesses, feedback) into structured constraint sets.

    Returns ``(known, must_include, must_not_at, excluded, min_count, max_count)``:

    - ``known[i]`` is the green-locked letter at position ``i``, or ``None``.
    - ``must_include`` is the set of letters proven to appear in the target
      (any letter with ``min_count[letter] >= 1``).
    - ``must_not_at[i]`` is the set of letters that cannot appear at position
      ``i`` (yellows at ``i``, grays at ``i`` of letters that still may appear
      elsewhere).
    - ``excluded`` is the set of letters proven absent from the target (i.e.
      letters with ``max_count[letter] == 0``).
    - ``min_count[letter]`` is the lower bound on how many times ``letter``
      appears in the target (0 if not observed).
    - ``max_count[letter]`` is the upper bound (5 if no gray has capped it).

    Wordle's duplicate-letter rule (used to compute the counts):

    Within a single guess, for each letter ``L`` independently, let ``gr`` be
    the number of GREEN tiles of ``L``, ``ye`` the number of YELLOW tiles of
    ``L``, and ``gy`` the number of GRAY tiles of ``L``. Then:

    * The target contains ``>= gr + ye`` copies of ``L`` (this guess proves
      at least that many).
    * If ``gy > 0``, the target contains *exactly* ``gr + ye`` copies of
      ``L`` — no more, no fewer — because Wordle only paints extra copies
      GRAY after all "real" copies have been assigned GREEN or YELLOW. In
      particular ``gr + ye == 0`` together with ``gy > 0`` means ``L`` is
      absent from the target (max_count[L] = 0).

    Aggregating across guesses: ``min_count[L]`` is the max over per-guess
    ``gr + ye`` observations; ``max_count[L]`` is the min over per-guess
    caps imposed by GRAY observations (no cap if no guess has GRAY-ed ``L``).
    """
    known: List[Optional[str]] = [None] * 5
    must_not_at: List[Set[str]] = [set() for _ in range(5)]
    min_count: Dict[str, int] = {}
    max_count: Dict[str, int] = {}

    for guess, fb in zip(previous_guesses, feedback_history):
        if not fb or fb == "Invalid guess":
            continue
        parts = fb.split()
        guess_gr: Counter = Counter()
        guess_ye: Counter = Counter()
        guess_gy: Counter = Counter()
        for pos, tok in enumerate(parts):
            if ":" not in tok or pos >= 5:
                continue
            letter, color = tok.split(":", 1)
            letter = letter.strip().upper()
            color = color.strip().upper()
            if len(letter) != 1:
                continue
            if color == "GREEN":
                known[pos] = letter
                guess_gr[letter] += 1
            elif color == "YELLOW":
                must_not_at[pos].add(letter)
                guess_ye[letter] += 1
            elif color == "GRAY":
                # The letter cannot be at this exact position; whether it
                # appears anywhere else is determined by min/max count below.
                must_not_at[pos].add(letter)
                guess_gy[letter] += 1

        for L in set(guess_gr) | set(guess_ye) | set(guess_gy):
            gr = guess_gr[L]
            ye = guess_ye[L]
            gy = guess_gy[L]
            # Lower bound: this guess proved the target contains >= gr+ye of L.
            if gr + ye > min_count.get(L, 0):
                min_count[L] = gr + ye
            # Upper bound: a gray on L in *this* guess caps total count at gr+ye.
            if gy > 0:
                cap = gr + ye
                prev_cap = max_count.get(L, 5)
                if cap < prev_cap:
                    max_count[L] = cap

    must_include: Set[str] = {L for L, c in min_count.items() if c >= 1}
    excluded: Set[str] = {L for L, c in max_count.items() if c == 0}
    # Defensive: a letter cannot be both proven-present and proven-absent;
    # contradictory inputs (e.g. malformed feedback) resolve in favor of
    # "proven present" since min_count is the stronger positive evidence.
    excluded -= must_include
    return known, must_include, must_not_at, excluded, min_count, max_count


def build_constraint_summary(
    previous_guesses: List[str],
    feedback_history: List[str],
) -> str:
    """Summarize constraints implied by all past guesses for the LM prompt.

    Feedback tokens look like: ``"C:GREEN A:YELLOW R:GRAY ..."`` in order of
    positions. Produces a multi-line summary covering:

    * ``Known pattern (position-locked greens): C??R?``
    * ``Must still include (yellow somewhere): A, R``
    * ``Must appear at least N times: P×2`` — emitted only when a letter's
      ``min_count`` is ``>= 2`` (duplicate-letter lower bound). Silent for
      ``min_count == 1`` since that case is already covered by the "must
      include" line.
    * ``Appears at most N times: E×1`` — emitted only when a gray has capped
      a letter that still has ``min_count >= 1`` (duplicate-letter upper
      bound that isn't the trivial "not in target"). For ``max_count == 0``
      the letter shows up under "Letters ruled out" instead.
    * ``Letters ruled out at least once (gray): K, X`` — letters with
      ``max_count == 0`` (guaranteed absent from the target).
    """
    if not previous_guesses or not feedback_history:
        return ""

    known, must_include, _must_not_at, excluded, min_count, max_count = (
        _parse_constraints(previous_guesses, feedback_history)
    )

    # A letter fully determined by greens (appears exactly min_count times,
    # all as greens) shouldn't also be listed as "must still include yellow
    # somewhere" — the position-locked line already covers it.
    green_counts: Counter = Counter(c for c in known if c is not None)
    include_outside_greens = {
        L for L in must_include if min_count.get(L, 0) > green_counts.get(L, 0)
    }

    lines: List[str] = []
    if any(c is not None for c in known):
        pattern = "".join((c if c else "?") for c in known)
        lines.append(f"Known pattern (position-locked greens): {pattern}")
    if include_outside_greens:
        lines.append(
            "Must still include (yellow somewhere): "
            + ", ".join(sorted(include_outside_greens))
        )

    # Duplicate-letter lower bounds: only emit when >= 2, to avoid restating
    # the single "must include" row above.
    multi_min = sorted((L, c) for L, c in min_count.items() if c >= 2)
    if multi_min:
        lines.append(
            "Must appear at least N times: "
            + ", ".join(f"{L}\u00d7{c}" for L, c in multi_min)
        )

    # Duplicate-letter upper bounds: emit only for letters that are present
    # AND capped (otherwise the "ruled out" line below covers the case).
    capped = sorted(
        (L, max_count[L])
        for L in max_count
        if max_count[L] >= 1 and min_count.get(L, 0) >= 1
    )
    if capped:
        lines.append(
            "Appears at most N times: "
            + ", ".join(f"{L}\u00d7{c}" for L, c in capped)
        )

    if excluded:
        lines.append(
            "Letters ruled out at least once (gray): " + ", ".join(sorted(excluded))
        )

    if not lines:
        return ""
    return "Constraint summary:\n" + "\n".join(lines)


def find_consistent_words(
    candidates: Iterable[str],
    previous_guesses: Sequence[str],
    feedback_history: Sequence[str],
) -> List[str]:
    """Filter ``candidates`` to those consistent with the observed constraints.

    Enforces Wordle's full duplicate-letter semantics via the per-letter
    ``min_count`` / ``max_count`` bounds from ``_parse_constraints``. A
    candidate ``u`` passes iff:

    * every green position matches (``u[i] == known[i]``),
    * no position-level ban fires (``u[i] not in must_not_at[i]``),
    * for every observed letter ``L``:
      ``min_count[L] <= u.count(L) <= max_count[L]``.

    The ``must_include`` / ``excluded`` sets from the parser are redundant
    with the count check (min>=1 and max==0 respectively) and are not used
    here, but are kept in the return tuple for callers who want the simpler
    summary view.

    With no prior guesses/feedback, returns ``list(candidates)`` (uppercased,
    5-letter alphabetic only) unchanged.
    """
    cleaned = [
        w.upper() for w in candidates if isinstance(w, str) and len(w) == 5 and w.isalpha()
    ]
    if not previous_guesses or not feedback_history:
        return cleaned

    known, _must_include, must_not_at, _excluded, min_count, max_count = (
        _parse_constraints(previous_guesses, feedback_history)
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
        if any(u[i] in banned for i, banned in enumerate(must_not_at)):
            continue
        counts = Counter(u)
        bad_count = False
        for L, lo in min_count.items():
            if counts.get(L, 0) < lo:
                bad_count = True
                break
        if bad_count:
            continue
        for L, hi in max_count.items():
            if counts.get(L, 0) > hi:
                bad_count = True
                break
        if bad_count:
            continue
        out.append(u)
    return out
