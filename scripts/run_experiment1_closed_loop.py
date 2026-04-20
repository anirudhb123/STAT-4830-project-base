#!/usr/bin/env python3
"""
Experiment 1 — Closed-loop proof of the bottleneck fix.

Mirror of `notebooks/week12_implementation_LoRARun.ipynb` cells 2 + 4 + 12 with
the EXPERIMENT_1_OVERRIDE applied (VOCAB_SCHEDULE=[4], N_ITERATIONS=30). Skips
the Phase-A probe cell (10). Designed so we can produce the experiment-1
diagnostics + verdict without depending on an interactive Jupyter kernel.

Per-stage diagnostics captured in the printed summary:
  - pre-warm-start success      (greedy quick_eval on the stage's secret pool)
  - post-warm-start success     -> ws_gain
  - post-ES success             -> es_gain
  - per-iter cos(ĝ), dprobe, popσ, wins/N (printed inline by train_es_wordle)

Usage (from repo root):
    .venv/bin/python scripts/run_experiment1_closed_loop.py 2>&1 | tee /tmp/exp1.log

Set EXP1_N_ITERATIONS=10 (or similar) to short-circuit for a smoke test.
"""
from __future__ import annotations

import gc
import importlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordle_env import (  # noqa: E402
    MOCK_WORDLE_TARGETS,
    load_wordle_environment,
)
from wordle_gpt2_policy import WordleGPT2Policy, TRANSFORMERS_AVAILABLE  # noqa: E402
from wordle_gpt2_warmstart import (  # noqa: E402
    quick_eval_success,
    supervised_warm_start_wordle,
)

import es_wordle  # noqa: E402

importlib.reload(es_wordle)
from es_wordle import (  # noqa: E402
    es_gradient_estimate_wordle,
    train_curriculum,
)

if not TRANSFORMERS_AVAILABLE:
    raise ImportError("Install transformers: pip install transformers")


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_chat_template_support() -> None:
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "Chat-template models require jinja2>=3.1.0."
        ) from exc
    installed = getattr(jinja2, "__version__", "0")
    if _parse_version_tuple(installed) < (3, 1, 0):
        raise ImportError(
            f"Chat-template models require jinja2>=3.1.0, found {installed}."
        )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_model_load_kwargs(device: torch.device) -> tuple[dict, str]:
    if device.type != "cuda":
        return {}, "float32"
    if torch.cuda.is_bf16_supported():
        return {"torch_dtype": torch.bfloat16}, "bfloat16"
    return {"torch_dtype": torch.float16}, "float16"


def main() -> None:
    # --- Mirror notebook cell 4 (gemma_full + EXPERIMENT_1_OVERRIDE) ---
    SEED = 42
    DEVICE = choose_device()
    MOCK_ENV = False
    USE_PRIME_TARGETS = True
    USE_LORA = True
    LORA_R = 8
    RICHER_PROMPT = True
    WARM_START_LR = 3e-4
    SIGMA = 0.02
    ALPHA = None  # auto-calibrate below
    NORMALIZE_GRADIENT = False
    RANK_FITNESS = True
    BASELINE_SUBTRACT = True
    ANTITHETIC = True
    COMMON_RANDOM_NUMBERS = True
    EMA_BETA = 0.0
    EVAL_DETERMINISTIC = True
    FITNESS_OBJECTIVE = "win_plus_return"
    WIN_FITNESS_SCALE = 8.0

    MODEL_NAME = "google/gemma-3-1b-it"
    USE_CHAT_TEMPLATE = True
    CHAT_GENERATION_PROMPT = True
    MAX_PROMPT_LENGTH = 512
    NUM_TRAIN_EXAMPLES = 2000
    NUM_EVAL_EXAMPLES = 20
    N_EVAL_EPISODES = 16
    EVAL_N_EPISODES = 50
    EVAL_EVERY = 1
    WARM_START_STEPS = 200
    N_POP = 64
    WARM_START_FEEDBACK_CONSISTENT = True

    # === EXPERIMENT_1_OVERRIDE (matches notebook cell 4) ===
    VOCAB_SCHEDULE = [4]
    N_ITERATIONS = int(os.environ.get("EXP1_N_ITERATIONS", "30"))
    HOLDOUT_MODE = "episode"
    SECRET_HOLDOUT_FRAC = 0.2  # ignored in episode mode
    MASKED_EVAL_SANITY_PROBE = True
    # Keep the head wide so the mock-targets assertion below + the policy
    # geometry match Test B (narrow secret pool, wide action space).
    MAX_VOCAB = 1024

    SECRET_POOL_SIZES = list(VOCAB_SCHEDULE)
    EVAL_POOL_SIZES = [0 for _ in SECRET_POOL_SIZES]
    WS_POOL_SIZES = list(SECRET_POOL_SIZES)
    # See cell 4 EXPERIMENT_1_OVERRIDE comment: WARM_START_STEPS=200 saturates
    # the 4-secret pool to 100% post-WS, forcing es_gain == 0 by construction.
    # Cut to 20 episodes (~2-3 opt steps at bs=8) so post-WS lands mid-curve.
    WARM_START_STEPS_PER_STAGE = [
        int(os.environ.get("EXP1_WARM_START_STEPS", "20"))
    ]

    require_chat_template_support()
    MODEL_LOAD_KWARGS, dtype_name = default_model_load_kwargs(DEVICE)

    print(f"ROOT: {ROOT}")
    print(
        f"device={DEVICE} dtype={dtype_name} model={MODEL_NAME} LoRA r={LORA_R} "
        f"max_vocab={MAX_VOCAB}"
    )
    print(
        f"=== EXPERIMENT 1: closed-loop proof | VOCAB_SCHEDULE={VOCAB_SCHEDULE} "
        f"N_ITERATIONS={N_ITERATIONS} N_POP={N_POP} n_eval_episodes={N_EVAL_EPISODES} "
        f"baseline_subtract={BASELINE_SUBTRACT} ema_beta={EMA_BETA} ==="
    )
    print(f"WARM_START_STEPS_PER_STAGE = {WARM_START_STEPS_PER_STAGE}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = DEVICE

    t0 = time.time()
    policy = WordleGPT2Policy(
        model_name=MODEL_NAME,
        use_prime_targets=USE_PRIME_TARGETS,
        max_vocab_size=MAX_VOCAB,
        max_prompt_length=MAX_PROMPT_LENGTH,
        include_mock_targets_in_vocab=True,
        richer_prompt=RICHER_PROMPT,
        use_chat_template=USE_CHAT_TEMPLATE,
        chat_generation_prompt=CHAT_GENERATION_PROMPT,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        model_kwargs=MODEL_LOAD_KWARGS,
    ).to(device)
    print(
        f"Policy built in {time.time() - t0:.1f}s | "
        f"trainable params: {policy.count_trainable_parameters():,}"
    )

    assert all(w in policy.words for w in MOCK_WORDLE_TARGETS), (
        "MOCK_WORDLE_TARGETS missing from policy.words; ES would generate target "
        "words the policy cannot emit."
    )

    env_train = load_wordle_environment(
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_eval_examples=NUM_EVAL_EXAMPLES,
        use_prime_intellect=not MOCK_ENV,
        target_pool=policy.words,
    )
    env_eval = load_wordle_environment(
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_eval_examples=NUM_EVAL_EXAMPLES,
        use_prime_intellect=not MOCK_ENV,
        target_pool=policy.words,
    )

    # --- Pre-warm-start eval on the stage's exact secret pool -------------------
    # train_curriculum installs the stage pool inside its own loop, but for the
    # "headline" pre-warm-start metric we want to measure on the same 4 secrets
    # ES will see, with greedy actions. Manually install the pool, eval, restore.
    stage_pool = list(policy.words[: VOCAB_SCHEDULE[0]])
    print(f"Stage 1 secret pool ({len(stage_pool)} words): {stage_pool}")
    env_train.set_target_pool(stage_pool)
    env_eval.set_target_pool(stage_pool)

    pre_ws_seed = SEED + 1
    rng_snap_py = random.getstate()
    rng_snap_np = np.random.get_state()
    rng_snap_torch = torch.get_rng_state()
    rng_snap_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        random.seed(pre_ws_seed)
        np.random.seed(pre_ws_seed)
        torch.manual_seed(pre_ws_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pre_ws_seed)
        pre_ws_indist = float(
            quick_eval_success(
                policy,
                env_train,
                n_episodes=EVAL_N_EPISODES,
                stochastic=False,
                max_turns=6,
            )
        )
    finally:
        random.setstate(rng_snap_py)
        np.random.set_state(rng_snap_np)
        torch.set_rng_state(rng_snap_torch)
        if rng_snap_cuda is not None:
            torch.cuda.set_rng_state_all(rng_snap_cuda)
    print(f"\n[pre-warm-start] greedy eval_success on stage pool: {pre_ws_indist:.1%}")

    # --- ALPHA calibration: one-shot gradient probe on the stage pool -----------
    # Same logic as cell 12's calibration. We need ALPHA before train_curriculum.
    PROBE_TARGET_STEP = 0.13
    cal_seed = SEED + 2
    rng_snap_py = random.getstate()
    rng_snap_np = np.random.get_state()
    rng_snap_torch = torch.get_rng_state()
    rng_snap_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        random.seed(cal_seed)
        np.random.seed(cal_seed)
        torch.manual_seed(cal_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cal_seed)
        _g, _af, _fits, _aw, _ps, _wrs = es_gradient_estimate_wordle(
            policy,
            env_train,
            N=N_POP,
            sigma=SIGMA,
            n_eval_episodes=N_EVAL_EPISODES,
            max_turns=6,
            rank_fitness=RANK_FITNESS,
            fitness_objective=FITNESS_OBJECTIVE,
            win_fitness_scale=WIN_FITNESS_SCALE,
            antithetic=ANTITHETIC,
            common_random_numbers=COMMON_RANDOM_NUMBERS,
            baseline_subtract=BASELINE_SUBTRACT,
        )
        _g_norm = float(_g.norm().item())
    finally:
        random.setstate(rng_snap_py)
        np.random.set_state(rng_snap_np)
        torch.set_rng_state(rng_snap_torch)
        if rng_snap_cuda is not None:
            torch.cuda.set_rng_state_all(rng_snap_cuda)
        del _g, _af, _fits, _aw, _ps, _wrs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if _g_norm > 1e-8:
        ALPHA = float(PROBE_TARGET_STEP / _g_norm)
        print(
            f"[ALPHA cal] raw ‖ĝ‖ = {_g_norm:.4g} -> "
            f"ALPHA = {ALPHA:.2e} (target step ≈ {PROBE_TARGET_STEP})"
        )
    else:
        ALPHA = 1e-5
        print(
            f"[ALPHA cal] ‖ĝ‖ ≈ 0; falling back to ALPHA={ALPHA:.2e}"
        )

    # train_curriculum will re-install the stage pool itself, so its env state
    # at start-of-stage matches what we just measured.
    history = train_curriculum(
        policy,
        env_train,
        vocab_schedule=VOCAB_SCHEDULE,
        n_iterations_per_stage=N_ITERATIONS,
        expand_action_space=False,
        env_eval=env_eval,
        holdout_mode=HOLDOUT_MODE,
        secret_holdout_frac=SECRET_HOLDOUT_FRAC,
        masked_eval_sanity_probe=MASKED_EVAL_SANITY_PROBE,
        warm_start_fn=supervised_warm_start_wordle,
        warm_start_steps=WARM_START_STEPS_PER_STAGE,
        warm_start_max_post_ws_success=0.85,
        warm_start_kwargs={
            "lr": WARM_START_LR,
            "device": device,
            "seed": SEED,
            "verbose": True,
            "batch_size": 8,
            "feedback_consistent_random": WARM_START_FEEDBACK_CONSISTENT,
        },
        post_warm_start_eval_episodes=EVAL_N_EPISODES,
        post_warm_start_eval_deterministic=EVAL_DETERMINISTIC,
        N=N_POP,
        sigma=SIGMA,
        alpha=ALPHA,
        n_eval_episodes=N_EVAL_EPISODES,
        max_turns=6,
        eval_every=EVAL_EVERY,
        verbose=True,
        normalize_gradient=NORMALIZE_GRADIENT,
        eval_n_episodes=EVAL_N_EPISODES,
        rank_fitness=RANK_FITNESS,
        eval_deterministic=EVAL_DETERMINISTIC,
        fitness_objective=FITNESS_OBJECTIVE,
        win_fitness_scale=WIN_FITNESS_SCALE,
        antithetic=ANTITHETIC,
        common_random_numbers=COMMON_RANDOM_NUMBERS,
        ema_beta=EMA_BETA,
        baseline_subtract=BASELINE_SUBTRACT,
    )

    # --- Per-iter table + verdict ----------------------------------------------
    train_iters = list(history.get("train_iter", []))
    train_cos = list(history.get("train_grad_cos", []))
    train_pop_std = list(history.get("pop_fitness_std", []))
    train_wins = list(history.get("train_win_count", []))
    train_dprobe = list(history.get("train_probe_delta", []))
    train_grad_norm = list(history.get("train_grad_norm", []))

    print("\n=== Per-iter ES diagnostics ===")
    print(
        f"  {'iter':>4} {'cos':>7} {'popσ':>9} {'wins/N':>7} "
        f"{'dprobe':>8} {'grad_norm':>10}"
    )
    for it, c, p, w, d, g in zip(
        train_iters, train_cos, train_pop_std, train_wins, train_dprobe, train_grad_norm
    ):
        c_str = "    n/a" if c != c else f"{c:+7.3f}"
        d_str = "    n/a " if d != d else f"{d:+7.1%}"
        print(
            f"  {it:>4} {c_str:>7} {p:>9.4f} {w:>4d}/{N_POP} {d_str:>8} {g:>10.2f}"
        )

    eval_iters = list(history.get("iteration", []))
    eval_succs = list(history.get("eval_success", []))
    print("\n=== Per-iter eval rollouts (50 episodes, greedy) ===")
    for it, s in zip(eval_iters, eval_succs):
        print(f"  iter={it:>4}  eval_success={s:.1%}")

    post_ws_indist = float(history.get("post_warmstart_success_indist", [float("nan")])[0])
    end_es = float(eval_succs[-1]) if eval_succs else float("nan")

    ws_gain = post_ws_indist - pre_ws_indist
    es_gain = end_es - post_ws_indist

    nonzero_dprobe = [d for d in train_dprobe if d == d and abs(d) > 1e-9]
    eval_dprobe_total = sum(1 for d in train_dprobe if d == d)
    nonzero_frac = len(nonzero_dprobe) / max(1, eval_dprobe_total)

    print("\n=== EXPERIMENT 1 SUMMARY ===")
    print(f"  pre-warm-start success      : {pre_ws_indist:.1%}")
    print(f"  post-warm-start success     : {post_ws_indist:.1%}   (ws_gain = {ws_gain:+.1%})")
    print(f"  post-ES success (final)     : {end_es:.1%}   (es_gain = {es_gain:+.1%})")
    print(
        f"  dprobe non-zero fraction    : {len(nonzero_dprobe)}/{eval_dprobe_total} = "
        f"{nonzero_frac:.0%}"
    )
    if eval_succs:
        peak_es = max(eval_succs)
        print(f"  peak eval_success (any iter): {peak_es:.1%}")

    pass_es_gain = (es_gain == es_gain) and (es_gain > 0.0)
    pass_dprobe = nonzero_frac >= 0.25
    pass_lift_10 = (es_gain == es_gain) and (es_gain >= 0.10)

    print(
        f"\n  PASS criteria:\n"
        f"    es_gain > 0          -> {pass_es_gain}\n"
        f"    es_gain >= +10pp     -> {pass_lift_10}\n"
        f"    dprobe non-zero >=25%-> {pass_dprobe}"
    )

    if pass_es_gain:
        print("\n  VERDICT: PASS (es_gain > 0). Proceed to Experiment 2.")
    else:
        print("\n  VERDICT: FAIL (es_gain <= 0). Do NOT proceed to Experiment 2 — diagnose first.")


if __name__ == "__main__":
    main()
