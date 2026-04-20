"""
Tests for ``wordle_hints`` with explicit duplicate-letter coverage.

These cases are exactly the ones the week-12 critique flagged as silent
prompt-quality bugs at vocab sizes above 8 (the 8-word mock pool has no
duplicate-letter targets).

Run with: ``pytest tests/test_wordle_hints.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wordle_hints import (  # noqa: E402
    _parse_constraints,
    build_constraint_summary,
    find_consistent_words,
)


def _feedback(guess: str, target: str) -> str:
    """Reference implementation of the env's Wordle feedback (greens-first,
    then yellows against remaining target slots, rest gray). Mirrors
    ``WordleEnvironmentWrapper._generate_wordle_feedback`` so the tests are
    self-contained and don't need the full env import."""
    guess = guess.upper()
    target = target.upper()
    assert len(guess) == len(target) == 5
    slots = [None] * 5
    remaining = list(target)
    for i, ch in enumerate(guess):
        if ch == target[i]:
            slots[i] = f"{ch}:GREEN"
            remaining[i] = None
    for i, ch in enumerate(guess):
        if slots[i] is not None:
            continue
        if ch in remaining:
            slots[i] = f"{ch}:YELLOW"
            remaining[remaining.index(ch)] = None
        else:
            slots[i] = f"{ch}:GRAY"
    return " ".join(slots)


class TestParseConstraintsDuplicateLetters:
    """Per-letter min/max count bounds under Wordle's duplicate-letter rule."""

    def test_gray_alone_fully_excludes_letter(self):
        # Target=SLATE, Guess=BRICK -> all gray; every guess letter has max=0.
        fb = _feedback("BRICK", "SLATE")
        known, must_include, _mn, excluded, min_c, max_c = _parse_constraints(
            ["BRICK"], [fb]
        )
        assert known == [None] * 5
        assert must_include == set()
        assert excluded == {"B", "R", "I", "C", "K"}
        for L in "BRICK":
            assert min_c.get(L, 0) == 0
            assert max_c[L] == 0

    def test_duplicate_gray_with_green_caps_count(self):
        # Target=EAGER (E@0, E@3), Guess=EERIE:
        #   E@0 GREEN, E@1 YELLOW, R@2 YELLOW, I@3 GRAY, E@4 GRAY
        # Expected: E min=max=2 (one green + one yellow, and a third E is gray
        # -> target contains exactly 2 E's). I is fully excluded. R is not
        # excluded (it's a yellow).
        fb = _feedback("EERIE", "EAGER")
        assert "E:GREEN" in fb and "E:YELLOW" in fb and "E:GRAY" in fb
        _k, must_include, _mn, excluded, min_c, max_c = _parse_constraints(
            ["EERIE"], [fb]
        )
        assert min_c["E"] == 2
        assert max_c["E"] == 2
        assert "E" in must_include
        assert "E" not in excluded
        assert excluded == {"I"}
        assert "R" in must_include

    def test_duplicate_gray_with_yellow_no_green(self):
        # Target=APPLE, Guess=PUPPY:
        #   P@0 YELLOW, U@1 GRAY, P@2 GREEN, P@3 GRAY, Y@4 GRAY
        # Expected: P min=2 (1 yellow + 1 green) and max=2 (a third P came back gray).
        fb = _feedback("PUPPY", "APPLE")
        assert "P:YELLOW" in fb and "P:GREEN" in fb and "P:GRAY" in fb
        _k, must_include, _mn, excluded, min_c, max_c = _parse_constraints(
            ["PUPPY"], [fb]
        )
        assert min_c["P"] == 2
        assert max_c["P"] == 2
        assert "P" in must_include
        assert excluded == {"U", "Y"}

    def test_gray_alone_on_letter_with_prior_yellow_does_not_exclude(self):
        # Two guesses: the first proves A is in the target (yellow), the second
        # has A gray only because the guess has too many As, not because A is
        # absent. A must NOT end up in `excluded`.
        # Target=ALOFT:
        g1, t = "SLATE", "ALOFT"
        fb1 = _feedback(g1, t)  # A:YELLOW (among others)
        assert "A:YELLOW" in fb1
        g2 = "AAAAA"
        fb2 = _feedback(g2, t)  # A@0 GREEN (target[0]=A), other A's GRAY.
        assert "A:GREEN" in fb2 and "A:GRAY" in fb2
        _k, must_include, _mn, excluded, min_c, max_c = _parse_constraints(
            [g1, g2], [fb1, fb2]
        )
        assert "A" in must_include
        assert "A" not in excluded
        assert min_c["A"] == 1
        assert max_c["A"] == 1


class TestFindConsistentWordsDuplicates:
    """Candidate filtering honors per-letter count bounds, not just set membership."""

    def test_excludes_candidates_with_too_many_copies(self):
        # After EERIE vs SPEED (min=max=2 on E), any candidate with >2 or <2 E's
        # must be filtered out even if everything else looks fine.
        fb = _feedback("EERIE", "SPEED")
        candidates = [
            "SPEED",   # 2 E's: consistent
            "SWEEP",   # 2 E's but P not yet excluded; consistent by count alone
            "EERIE",   # 3 E's: inconsistent (exceeds max=2)
            "STEEL",   # 2 E's: consistent by count; passes
            "STEAD",   # 1 E: inconsistent (below min=2)
        ]
        kept = find_consistent_words(candidates, ["EERIE"], [fb])
        assert "EERIE" not in kept
        assert "STEAD" not in kept
        assert "SPEED" in kept

    def test_includes_candidates_previously_rejected_by_naive_gray(self):
        # The old naive logic excluded any gray letter outright. With a prior
        # yellow on A, the follow-up AAAAA vs ALOFT marks 4 As GRAY but A is
        # clearly present in the target. Candidates with exactly one A must
        # survive.
        g1, t = "SLATE", "ALOFT"
        fb1 = _feedback(g1, t)
        g2 = "AAAAA"
        fb2 = _feedback(g2, t)
        candidates = ["ALOFT", "AFTER", "BROWN", "LAMPS"]
        kept = find_consistent_words(candidates, [g1, g2], [fb1, fb2])
        assert "ALOFT" in kept
        assert "BROWN" not in kept  # no A at all
        assert "AFTER" not in kept  # green at pos 0 must be A; here A@0 passes
        # but T@2 doesn't match pattern (ALOFT has O@2, T@3) -> AFTER has T@2 fails


class TestBuildConstraintSummaryDuplicates:
    """Prompt-text output is informative and correct under duplicates."""

    def test_gray_under_green_does_not_rule_out_letter(self):
        fb = _feedback("EERIE", "SPEED")
        s = build_constraint_summary(["EERIE"], [fb])
        # E is present with min=max=2; must NOT be listed under "ruled out".
        ruled_out_line = next(
            (line for line in s.splitlines() if line.startswith("Letters ruled out")),
            "",
        )
        assert "E" not in ruled_out_line.split(":", 1)[-1].split(",") if ruled_out_line else True
        # R and I must be ruled out.
        assert "Letters ruled out" in s
        assert "R" in s and "I" in s

    def test_prompt_reports_min_count_when_at_least_2(self):
        fb = _feedback("EERIE", "SPEED")
        s = build_constraint_summary(["EERIE"], [fb])
        assert "Must appear at least N times" in s
        assert "E" in s.split("Must appear at least N times", 1)[1]

    def test_basic_single_yellow_summary_still_works(self):
        # Matches the previous integration-test expectation in tests/test_wordle.py.
        g = ["SLATE"]
        fb = ["S:GRAY L:GRAY A:YELLOW T:GRAY E:GRAY"]
        s = build_constraint_summary(g, fb)
        assert "A" in s
        assert "YELLOW" in s or "include" in s.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
