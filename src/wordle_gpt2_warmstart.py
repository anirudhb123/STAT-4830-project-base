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
except ImportError:
    from wordle_gpt2_policy import WordleGPT2Policy


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
) -> Dict[str, List[float]]:
    """
    Sample mock episodes, play 1–4 random valid guesses, then train the policy to predict
    the secret word (index) from the resulting prompt. The secret never appears in the text.

    If ``exclude_target_from_random_guesses`` is True, random guesses never pick the secret,
    so episodes rarely end before the supervised step (fewer wasted skips).

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

    opt = torch.optim.Adam(trainable, lr=lr)
    policy.train()

    losses: List[float] = []
    skipped = 0

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
            candidates = [w for w in policy.words if w not in state.previous_guesses]
            if exclude_target_from_random_guesses and target in candidates:
                candidates = [w for w in candidates if w != target]
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

        y = policy.word_to_idx[target]
        logits = policy.forward_logits(state)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([y], device=device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())

        if verbose and (step + 1) % max(1, n_steps // 5) == 0:
            print(f"  warm-start {step+1}/{n_steps} | loss={np.mean(losses[-50:]):.4f} | skipped={skipped}")

    policy.eval()
    return {"loss": losses, "skipped": skipped}


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
