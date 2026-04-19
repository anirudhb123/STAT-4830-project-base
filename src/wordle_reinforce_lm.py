"""
REINFORCE (Monte Carlo policy gradient) for Wordle with WordleGPT2Policy.

Trains the LM head (and optional LoRA) by backprop through sampled actions;
frozen backbone weights stay in no_grad inside the policy as implemented in
`wordle_gpt2_policy.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch

try:
    from tqdm.auto import tqdm as _tqdm_bar
except ImportError:
    _tqdm_bar = None  # type: ignore[misc, assignment]


def _mask_logits_for_guesses(
    logits: torch.Tensor, policy: Any, previous_guesses: Optional[List[str]]
) -> torch.Tensor:
    if not previous_guesses:
        return logits
    out = logits.clone()
    for g in previous_guesses:
        u = g.upper()
        if u in policy.word_to_idx:
            out[policy.word_to_idx[u]] = -float("inf")
    return out


def _action_xml(word: str, state: Any) -> str:
    turn = state.turn_number + 1
    if turn == 1:
        think = f"Using the language model prior, opening with {word}."
    else:
        think = f"Conditioning on feedback, next guess: {word}."
    return f"<redacted_thinking>{think}</redacted_thinking>\n<guess>{word}</guess>"


def _sample_action(
    policy: Any, state: Any, previous_guesses: Optional[List[str]]
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    logits = policy.forward_logits(state)
    logits = _mask_logits_for_guesses(logits, policy, previous_guesses)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    word = policy.idx_to_word[int(action.item())]
    return word, log_prob, entropy


def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    g = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        g = float(r) + gamma * g
        out.append(g)
    out.reverse()
    return out


def collect_episode_reinforce(
    policy: Any,
    env: Any,
    max_turns: int = 6,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], bool, Dict[str, Any]]:
    """
    One stochastic rollout. Returns per-step log_probs, entropies, rewards, win flag, last info.
    """
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    done = False
    turns = 0
    info: Dict[str, Any] = {}

    while not done and turns < max_turns:
        word, lp, ent = _sample_action(policy, state, state.previous_guesses)
        action_xml = _action_xml(word, state)
        state, reward, done, info = env.step(action_xml)
        log_probs.append(lp)
        entropies.append(ent)
        rewards.append(float(reward))
        turns += 1

    won = float(info.get("correct_answer", 0.0)) >= 0.5
    return log_probs, entropies, rewards, won, info


def evaluate_wordle_lm_policy(
    policy: Any,
    env: Any,
    n_episodes: int = 20,
    max_turns: int = 6,
    deterministic: bool = True,
) -> Tuple[float, float, float]:
    """Mean return, success rate, mean turns (for completed or max-turn games)."""
    policy.eval()
    total_r = 0.0
    wins = 0
    turns_list: List[int] = []

    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            ep_r = 0.0
            done = False
            turns = 0
            while not done and turns < max_turns:
                emb = env.get_state_embedding(state)
                if hasattr(policy, "format_action_xml"):
                    action_xml, _ = policy.format_action_xml(
                        state, emb, deterministic=deterministic
                    )
                else:
                    action_idx, _ = policy.get_action(emb, deterministic=deterministic)
                    word = policy.vocab.action_to_word(action_idx)
                    action_xml = f"<guess>{word}</guess>"
                state, reward, done, info = env.step(action_xml)
                ep_r += float(reward)
                turns += 1
            total_r += ep_r
            if float(info.get("correct_answer", 0.0)) >= 0.5:
                wins += 1
            turns_list.append(turns)

    mean_r = total_r / max(1, n_episodes)
    success = wins / max(1, n_episodes)
    mean_turns = float(np.mean(turns_list)) if turns_list else 0.0
    return mean_r, success, mean_turns


def train_reinforce_wordle(
    policy: Any,
    env: Any,
    optimizer: torch.optim.Optimizer,
    n_iterations: int = 100,
    n_episodes_per_iter: int = 8,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    max_turns: int = 6,
    eval_every: int = 5,
    eval_n_episodes: int = 24,
    normalize_advantages: bool = True,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, List[float]]:
    """
    REINFORCE with discounted per-step returns and optional entropy bonus.

    Advantages are per-step discounted returns minus the mean advantage in the
    update batch (variance reduction).

    If ``tqdm`` is installed and ``show_progress`` is True, shows a progress bar
    over training iterations (loss / mean train return / eval success when run).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    history: Dict[str, List] = {
        "loss": [],
        "mean_return": [],
        "eval_return": [],
        "eval_success": [],
        "eval_turns": [],
        # Align with ES `train_es_wordle` plotting: eval checkpoints + per-iter diagnostics
        "iteration": [],
        "train_iter": [],
        "param_drift": [],
        "batch_return_std": [],
    }

    def _trainable_flat() -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in policy.parameters() if p.requires_grad])

    params_init = _trainable_flat().clone()

    outer = range(n_iterations)
    if show_progress and _tqdm_bar is not None:
        outer = _tqdm_bar(outer, desc="REINFORCE", unit="iter")
    last_eval_suffix = ""
    warned_no_tqdm = False
    for it in outer:
        policy.train()
        batch_logp: List[torch.Tensor] = []
        batch_ent: List[torch.Tensor] = []
        batch_adv: List[float] = []
        iter_returns: List[float] = []

        for _ in range(n_episodes_per_iter):
            log_probs, entropies, rewards, _, _ = collect_episode_reinforce(
                policy, env, max_turns=max_turns
            )
            if not log_probs:
                continue
            returns = _discounted_returns(rewards, gamma)
            g_ep = float(sum(rewards))
            iter_returns.append(g_ep)

            adv = [float(x) for x in returns]
            for lp, ent, a in zip(log_probs, entropies, adv):
                batch_logp.append(lp)
                batch_ent.append(ent)
                batch_adv.append(a)

        if not batch_logp:
            continue

        adv_t = torch.tensor(batch_adv, device=batch_logp[0].device, dtype=torch.float32)
        if normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        loss_pi = 0.0
        loss_ent = 0.0
        for lp, ent, a in zip(batch_logp, batch_ent, adv_t):
            loss_pi = loss_pi - lp * a
            loss_ent = loss_ent - entropy_coef * ent
        n = len(batch_logp)
        loss = (loss_pi + loss_ent) / n

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["mean_return"].append(float(np.mean(iter_returns)) if iter_returns else 0.0)
        history["train_iter"].append(int(it))
        params_now = _trainable_flat()
        history["param_drift"].append(float((params_now - params_init).norm().item()))
        history["batch_return_std"].append(
            float(np.std(iter_returns)) if len(iter_returns) > 1 else 0.0
        )

        postfix: Dict[str, str] = {
            "loss": f"{float(loss.item()):.4f}",
            "mret": f"{history['mean_return'][-1]:.3f}",
        }
        if eval_every > 0 and (it + 1) % eval_every == 0:
            er, es, et = evaluate_wordle_lm_policy(
                policy,
                env,
                n_episodes=eval_n_episodes,
                max_turns=max_turns,
                deterministic=True,
            )
            history["eval_return"].append(er)
            history["eval_success"].append(es)
            history["eval_turns"].append(et)
            history["iteration"].append(int(it))
            last_eval_suffix = f"{es:.1%}"
        if last_eval_suffix:
            postfix["eval_succ"] = last_eval_suffix

        if show_progress and _tqdm_bar is not None and hasattr(outer, "set_postfix"):
            outer.set_postfix(**postfix)  # type: ignore[union-attr]
        elif show_progress and _tqdm_bar is None and not warned_no_tqdm:
            warned_no_tqdm = True
            warnings.warn(
                "Install tqdm for a training progress bar: pip install tqdm",
                stacklevel=1,
            )

    return history
