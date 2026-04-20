"""
Supervised warm-start for WordleGPT2Policy: cross-entropy toward the hidden target word
after random play (target is NOT in the prompt; only used as label).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .wordle_gpt2_policy import WordleGPT2Policy
    from .wordle_hints import find_consistent_words
except ImportError:
    from wordle_gpt2_policy import WordleGPT2Policy
    from wordle_hints import find_consistent_words


def supervised_warm_start_wordle(
    policy: WordleGPT2Policy,
    env: Any,
    n_steps: int = 400,
    lr: float = 3e-4,
    max_turns: int = 6,
    device: Optional[torch.device] = None,
    seed: int = 0,
    verbose: bool = True,
    min_random_guesses: int = 1,
    max_random_guesses: int = 4,
    exclude_target_from_random_guesses: bool = True,
    batch_size: int = 1,
    feedback_consistent_random: bool = False,
) -> Dict[str, List[float]]:
    """
    Sample mock episodes, play 1–4 random valid guesses, then train the policy to predict
    the secret word (index) from the resulting prompt. The secret never appears in the text.

    If ``exclude_target_from_random_guesses`` is True, random guesses never pick the secret,
    so episodes rarely end before the supervised step (fewer wasted skips).

    If ``feedback_consistent_random`` is True, after the first random guess each subsequent
    random guess is drawn from the subset of valid words consistent with the accumulated
    feedback (greens / yellows / grays via ``wordle_hints.find_consistent_words``). This
    makes the resulting (state, target) example a much sharper "given these constraints,
    predict the secret" signal: the supervised target is then almost always one of a small
    set of remaining-consistent words. With large vocabularies this is the difference between
    the head learning a meaningful conditional distribution and the head trying to memorize
    arbitrary (garbage prefix → random secret) pairs. Falls back to all unguessed words
    when no consistent candidate is left (so warm-start episodes don't stall).

    ``batch_size`` controls how many sampled (state, target) examples are accumulated per
    optimizer step. ``batch_size=1`` (default) reproduces the original one-example-per-step
    behavior. With ``batch_size>1``, the policy must expose ``forward_logits_batch``
    (``WordleGPT2Policy`` does); episodes are still sampled one at a time, but a single
    batched LM forward + backward is performed every ``batch_size`` valid examples. This
    reduces wall-clock dramatically for large LMs (e.g. Gemma-3-1B), at the cost of a
    smaller number of (effectively-larger) optimizer steps.

    ``n_steps`` always counts *sampled episodes*. The number of optimizer steps is
    therefore approximately ``(n_steps - skipped) / batch_size``.

    Requires trainable parameters (head and optionally LoRA).
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = next(policy.parameters()).device

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters; enable LoRA or ensure head requires_grad.")

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if batch_size > 1 and not hasattr(policy, "forward_logits_batch"):
        raise ValueError(
            "batch_size > 1 requires policy.forward_logits_batch (WordleGPT2Policy). "
            f"Got policy of type {type(policy).__name__} without that method."
        )

    opt = torch.optim.Adam(trainable, lr=lr)
    policy.train()

    losses: List[float] = []
    skipped = 0

    state_buffer: List[Any] = []
    target_buffer: List[int] = []
    opt_steps = 0

    def _flush_batch() -> None:
        nonlocal opt_steps
        if not state_buffer:
            return
        targets = torch.tensor(target_buffer, dtype=torch.long, device=device)
        if batch_size > 1 or len(state_buffer) > 1:
            logits = policy.forward_logits_batch(state_buffer)  # [B, action_dim]
        else:
            logits = policy.forward_logits(state_buffer[0]).unsqueeze(0)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())
        opt_steps += 1
        state_buffer.clear()
        target_buffer.clear()

    for step in range(n_steps):
        state = env.reset()
        target = (state.target_word or "").upper()
        if not target or target not in policy.word_to_idx:
            skipped += 1
            continue

        n_rand = rng.randint(min_random_guesses, max_random_guesses)
        done = False
        info: Dict[str, Any] = {"correct_answer": 0.0}
        for _ in range(n_rand):
            if state.turn_number >= max_turns - 1:
                break
            base_candidates = [w for w in policy.words if w not in state.previous_guesses]
            if exclude_target_from_random_guesses:
                base_candidates = [w for w in base_candidates if w != target]
            if feedback_consistent_random and state.previous_guesses:
                consistent = find_consistent_words(
                    base_candidates,
                    state.previous_guesses,
                    state.feedback_history,
                )
                # Fallback to the broader set when the constraint pool is empty
                # (can happen if the simplified-duplicate-letter logic is too
                # strict, or if every consistent word equals the excluded target).
                candidates = consistent if consistent else base_candidates
            else:
                candidates = base_candidates
            if not candidates:
                break
            guess = rng.choice(candidates)
            state, _, done, info = env.step(f"<guess>{guess}</guess>")
            if done:
                if float(info.get("correct_answer", 0.0)) >= 0.5:
                    skipped += 1
                break

        if done and float(info.get("correct_answer", 0.0)) >= 0.5:
            continue
        if not state.previous_guesses:
            skipped += 1
            continue

        state_buffer.append(state)
        target_buffer.append(policy.word_to_idx[target])

        if len(state_buffer) >= batch_size:
            _flush_batch()

        if verbose and (step + 1) % max(1, n_steps // 5) == 0 and losses:
            print(
                f"  warm-start {step+1}/{n_steps} | "
                f"loss={np.mean(losses[-50:]):.4f} | "
                f"opt_steps={opt_steps} | bs={batch_size} | "
                f"skipped={skipped}"
            )

    # Flush any remaining partial batch so no examples are silently dropped.
    _flush_batch()

    policy.eval()
    return {"loss": losses, "skipped": skipped, "opt_steps": opt_steps}


def _wrap_guess_xml(state: Any, word: str) -> str:
    """Replicate ``WordleGPT2Policy.format_action_xml``'s wrapper without re-running the LM.

    Mirrors the inline helper in ``es_wordle._format_action_xml_for_word`` so this module
    has no circular import on ``es_wordle``.
    """
    turn = state.turn_number + 1
    if turn == 1:
        think = f"Using the language model prior, opening with {word}."
    else:
        think = f"Conditioning on feedback, next guess: {word}."
    return f"<think>{think}</think>\n<guess>{word}</guess>"


def quick_eval_success(
    policy: WordleGPT2Policy,
    env: Any,
    n_episodes: int = 32,
    stochastic: bool = True,
    max_turns: int = 6,
    device: Optional[torch.device] = None,
    return_argmax_in_set: Optional[Iterable[int]] = None,
) -> Union[float, Tuple[float, float]]:
    """Fraction of mock episodes solved within ``max_turns`` (diagnostic).

    When ``return_argmax_in_set`` is provided, also tracks the fraction of *greedy*
    actions whose argmax index lies in that set, and returns ``(win_rate, in_set_frac)``.
    Useful for confirming that the head's argmax is concentrated on the in-vocab pool
    (or, for the original buggy metric, is *never* in the held-out suffix). Tracking is
    only meaningful in greedy mode (``stochastic=False``); under sampling the fraction
    is computed against the sampled action and is not strictly an "argmax" measurement.

    Default return type is ``float`` (legacy callers); the tuple form is opt-in.
    """
    if device is None:
        device = next(policy.parameters()).device
    track_set: Optional[Set[int]] = None
    if return_argmax_in_set is not None:
        track_set = set(int(i) for i in return_argmax_in_set)
    in_set_count = 0
    action_count = 0
    policy.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            turns = 0
            info: Dict[str, Any] = {"correct_answer": 0.0}
            while not done and turns < max_turns:
                emb = env.get_state_embedding(state)
                if track_set is None:
                    xml, _ = policy.format_action_xml(state, emb, deterministic=not stochastic)
                else:
                    # Manual path: get the chosen index so we can record whether it's in
                    # the tracked set, then format the XML with the same template
                    # `format_action_xml` would have used.
                    idx, _lp = policy.get_action(
                        emb,
                        deterministic=not stochastic,
                        previous_guesses=state.previous_guesses,
                        state=state,
                    )
                    if idx in track_set:
                        in_set_count += 1
                    action_count += 1
                    xml = _wrap_guess_xml(state, policy.idx_to_word[idx])
                state, _, done, info = env.step(xml)
                turns += 1
            if float(info.get("correct_answer", 0.0)) >= 0.5:
                wins += 1
    win_rate = wins / max(1, n_episodes)
    if track_set is None:
        return win_rate
    in_set_frac = (in_set_count / action_count) if action_count > 0 else float("nan")
    return win_rate, in_set_frac


def quick_eval_success_masked(
    policy: WordleGPT2Policy,
    env: Any,
    allowed_indices: Iterable[int],
    n_episodes: int = 32,
    stochastic: bool = False,
    max_turns: int = 6,
    device: Optional[torch.device] = None,
) -> float:
    """Fraction of mock episodes solved within ``max_turns`` when the head's argmax /
    sampling is restricted to ``allowed_indices`` (a subset of action indices).

    Implementation: for each step, compute full logits, mask non-allowed indices to
    ``-inf`` (in addition to previous-guess masking already done by
    ``policy.get_action``-style consumers), then take ``argmax`` (greedy) or sample.

    This is the diagnostic that proves the prior 0% on word-level holdout was a metric
    construction artifact: when the eval-pool indices were *never* CE targets nor ES
    winners, an unrestricted greedy argmax cannot land on them; restricting the argmax
    to those indices recovers a non-trivial success rate from the same checkpoint.

    In ``holdout_mode="episode"``, ``allowed_indices`` is typically the in-vocab stage
    pool, and the gap vs. the unmasked greedy eval quantifies how much argmax mass is
    leaking onto out-of-pool words.
    """
    if device is None:
        device = next(policy.parameters()).device
    allowed = set(int(i) for i in allowed_indices)
    if not allowed:
        return float("nan")
    action_dim = int(getattr(policy, "action_dim", 0)) or len(getattr(policy, "words", []))
    if action_dim <= 0:
        raise RuntimeError("Cannot determine policy.action_dim for masked eval.")
    word_to_idx = getattr(policy, "word_to_idx", {}) or {}
    idx_to_word = getattr(policy, "idx_to_word", None)
    if idx_to_word is None:
        raise RuntimeError("Policy does not expose idx_to_word; cannot run masked eval.")

    # Static disallowed mask: indices NOT in `allowed` are forbidden every step.
    disallow = torch.zeros(action_dim, dtype=torch.bool, device=device)
    disallow[:] = True
    for i in allowed:
        if 0 <= i < action_dim:
            disallow[i] = False

    policy.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            turns = 0
            info: Dict[str, Any] = {"correct_answer": 0.0}
            while not done and turns < max_turns:
                logits = policy.forward_logits(state).clone()
                # Mask previously-guessed words (matches `WordleGPT2Policy.get_action` semantics).
                for g in state.previous_guesses:
                    u = g.upper()
                    if u in word_to_idx:
                        logits[word_to_idx[u]] = float("-inf")
                logits[disallow] = float("-inf")
                if stochastic:
                    probs = torch.softmax(logits, dim=-1)
                    idx = int(torch.distributions.Categorical(probs=probs).sample().item())
                else:
                    idx = int(torch.argmax(logits).item())
                xml = _wrap_guess_xml(state, idx_to_word[idx])
                state, _, done, info = env.step(xml)
                turns += 1
            if float(info.get("correct_answer", 0.0)) >= 0.5:
                wins += 1
    return wins / max(1, n_episodes)
