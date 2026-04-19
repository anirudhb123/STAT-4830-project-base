"""
Supervised warm-start for WordleGPT2Policy: cross-entropy toward the hidden target word
after random play (target is NOT in the prompt; only used as label).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

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


def quick_eval_success(
    policy: WordleGPT2Policy,
    env: Any,
    n_episodes: int = 32,
    stochastic: bool = True,
    max_turns: int = 6,
    device: Optional[torch.device] = None,
) -> float:
    """Fraction of mock episodes solved within max_turns (diagnostic)."""
    if device is None:
        device = next(policy.parameters()).device
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
                xml, _ = policy.format_action_xml(state, emb, deterministic=not stochastic)
                state, _, done, info = env.step(xml)
                turns += 1
            if float(info.get("correct_answer", 0.0)) >= 0.5:
                wins += 1
    return wins / max(1, n_episodes)
