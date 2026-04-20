#!/usr/bin/env python3
"""
Experiment 2 — Mini-batch ES under CRN at VOCAB_SCHEDULE=[16].

Mirror of notebook cells 2 + 4 + 12 with ACTIVE_EXPERIMENT="exp2". Single-stage
probe of the week-12 bottleneck-fix hypothesis: restrict each ES iter to
``PER_ITER_SECRET_SUBSET_SIZE=4`` secrets drawn uniformly from the 16-word
pool, so each population member plays each subset secret ~4 times per iter
under CRN (matching Test B's working-regime signal density).

Pass criteria (from the experiment brief):
  - es_gain >= +10pp on the held-out 16-word eval slate
  - dprobe non-zero on >= 25% of eval iterations

We additionally report a STOCHASTIC greedy-vs-sampling comparison because
Exp 1 showed greedy can saturate while ES_win still climbs (metric artifact,
see docs/llm_exploration/week12_log.md). At vocab=16 greedy has more
headroom, but tracking both prevents a repeat silent failure.

Usage (from repo root):
    .venv/bin/python -u scripts/run_experiment2_minibatch_crn.py 2>&1 | \
        tee /tmp/exp2_full.log

Environment overrides (for smoke tests):
    EXP2_N_ITERATIONS=2          # number of ES iters
    EXP2_WARM_START_STEPS=200    # per-stage warm-start episode budget
    EXP2_SUBSET_SIZE=4           # k for per_iter_secret_subset_size
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


def _snapshot_and_reseed(seed: int) -> dict:
    """Save RNG state, then reseed all RNGs to ``seed``. Returns the snapshot
    so the caller can restore afterward via ``_restore_snapshot``."""
    snap = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return snap


def _restore_snapshot(snap: dict) -> None:
    random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.set_rng_state(snap["torch"])
    if snap["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["cuda"])


def _eval_greedy_and_stochastic(
    policy, env, eval_n_episodes: int, probe_seed: int
) -> tuple[float, float]:
    """Run quick_eval_success twice -- greedy and stochastic -- with RNG
    snapshot/restore so the measurement doesn't perturb outer state.
    Returns (greedy_success, stochastic_success)."""
    snap = _snapshot_and_reseed(probe_seed)
    try:
        g = float(
            quick_eval_success(
                policy,
                env,
                n_episodes=eval_n_episodes,
                stochastic=False,
                max_turns=6,
            )
        )
    finally:
        _restore_snapshot(snap)
    snap = _snapshot_and_reseed(probe_seed + 1000)
    try:
        s = float(
            quick_eval_success(
                policy,
                env,
                n_episodes=eval_n_episodes,
                stochastic=True,
                max_turns=6,
            )
        )
    finally:
        _restore_snapshot(snap)
    return g, s


def main() -> None:
    # --- Mirror notebook cell 4 (gemma_full base + ACTIVE_EXPERIMENT='exp2') ---
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

    # === ACTIVE_EXPERIMENT = 'exp2' ===
    VOCAB_SCHEDULE = [16]
    N_ITERATIONS = int(os.environ.get("EXP2_N_ITERATIONS", "30"))
    HOLDOUT_MODE = "episode"
    SECRET_HOLDOUT_FRAC = 0.2  # ignored in episode mode
    MASKED_EVAL_SANITY_PROBE = True
    # Keep action head wide (1024) so the mock-targets assertion holds and
    # the geometry matches gemma_full's stage-1 setup (wide head, narrow pool).
    MAX_VOCAB = 1024

    SECRET_POOL_SIZES = list(VOCAB_SCHEDULE)
    EVAL_POOL_SIZES = [0 for _ in SECRET_POOL_SIZES]
    WS_POOL_SIZES = list(SECRET_POOL_SIZES)
    # WARM_START_STEPS_PER_STAGE: default to 200 (matches gemma_full stage-1
    # budget at vocab=16). The train_curriculum warm_start_max_post_ws_success=
    # 0.85 ceiling is the runtime backstop. If post-WS still saturates at 100%
    # (Exp 1 metric failure mode repeating), cut this via EXP2_WARM_START_STEPS.
    WARM_START_STEPS_PER_STAGE = [
        int(os.environ.get("EXP2_WARM_START_STEPS", "200"))
    ]
    PER_ITER_SECRET_SUBSET_SIZE = int(os.environ.get("EXP2_SUBSET_SIZE", "4"))

    require_chat_template_support()
    MODEL_LOAD_KWARGS, dtype_name = default_model_load_kwargs(DEVICE)

    print(f"ROOT: {ROOT}")
    print(
        f"device={DEVICE} dtype={dtype_name} model={MODEL_NAME} LoRA r={LORA_R} "
        f"max_vocab={MAX_VOCAB}"
    )
    print(
        f"=== EXPERIMENT 2: mini-batch ES under CRN | VOCAB_SCHEDULE={VOCAB_SCHEDULE} "
        f"PER_ITER_SECRET_SUBSET_SIZE={PER_ITER_SECRET_SUBSET_SIZE} "
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

    # --- Pre-warm-start eval on the stage's full 16-word pool ------------------
    # ES will sample 4-word subsets of this pool per iter, but pre/post-WS and
    # final-ES metrics are measured on the full 16 so the verdict reflects
    # generalization, not the within-subset overfit.
    stage_pool = list(policy.words[: VOCAB_SCHEDULE[0]])
    print(f"Stage 1 secret pool ({len(stage_pool)} words): {stage_pool}")
    env_train.set_target_pool(stage_pool)
    env_eval.set_target_pool(stage_pool)

    pre_ws_greedy, pre_ws_stoch = _eval_greedy_and_stochastic(
        policy, env_train, EVAL_N_EPISODES, probe_seed=SEED + 1
    )
    print(
        f"\n[pre-warm-start] eval_success on 16-word stage pool: "
        f"greedy={pre_ws_greedy:.1%}  stochastic={pre_ws_stoch:.1%}"
    )

    # --- ALPHA calibration: one-shot gradient probe on the full stage pool ----
    # We calibrate on the FULL pool (not a 4-word subset) so ALPHA reflects the
    # scale of gradient norms ES will see once per_iter_secret_subset_size is
    # in effect. Within-iter, each call to es_gradient_estimate_wordle with
    # subset_size=4 gets a gradient over 4 secrets; the norm scales weakly
    # with pool composition, so a full-pool calibration is a reasonable proxy
    # (and matches cell 12's calibration-before-curriculum design).
    PROBE_TARGET_STEP = 0.13
    cal_seed = SEED + 2
    snap = _snapshot_and_reseed(cal_seed)
    try:
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
            per_iter_secret_subset_size=PER_ITER_SECRET_SUBSET_SIZE,
        )
        _g_norm = float(_g.norm().item())
    finally:
        _restore_snapshot(snap)
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

    # Optional post-calibration scale for the step-size/overshoot diagnostic.
    # Exp 2 (k=4 subset rotating each iter) showed the calibrated ALPHA produced
    # an average per-iter step of ~0.45 vs the 0.13 target -- overshoot from
    # iter-0 calibration's implicit stationary-objective assumption not holding
    # under rotating mini-batches. EXP2_ALPHA_SCALE=0.25 quarters ALPHA to put
    # the expected step back on the 0.13 target accumulated across 4 iters.
    alpha_scale = float(os.environ.get("EXP2_ALPHA_SCALE", "1.0"))
    if alpha_scale != 1.0:
        ALPHA = ALPHA * alpha_scale
        print(
            f"[ALPHA cal] post-calibration scale EXP2_ALPHA_SCALE={alpha_scale} -> "
            f"ALPHA = {ALPHA:.2e}"
        )

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
        per_iter_secret_subset_size=PER_ITER_SECRET_SUBSET_SIZE,
    )

    # --- Explicit post-run stochastic eval on the full 16-word pool ------------
    # history's eval_success is greedy. We also want stochastic to sanity-check
    # against the Exp 1 saturation trap.
    end_greedy, end_stoch = _eval_greedy_and_stochastic(
        policy, env_eval, EVAL_N_EPISODES, probe_seed=SEED + 9001
    )

    # --- Per-iter table + verdict ----------------------------------------------
    train_iters = list(history.get("train_iter", []))
    train_cos = list(history.get("train_grad_cos", []))
    train_pop_std = list(history.get("pop_fitness_std", []))
    train_wins = list(history.get("train_win_count", []))
    train_dprobe = list(history.get("train_probe_delta", []))
    train_grad_norm = list(history.get("train_grad_norm", []))
    train_eswin = list(history.get("train_es_win", []))

    print("\n=== Per-iter ES diagnostics ===")
    print(
        f"  {'iter':>4} {'cos':>7} {'popσ':>9} {'wins/N':>7} "
        f"{'ES_win':>7} {'dprobe':>8} {'grad_norm':>10}"
    )
    for it, c, p, w, ew, d, g in zip(
        train_iters, train_cos, train_pop_std, train_wins,
        train_eswin, train_dprobe, train_grad_norm,
    ):
        c_str = "    n/a" if c != c else f"{c:+7.3f}"
        d_str = "    n/a " if d != d else f"{d:+7.1%}"
        ew_str = "    n/a" if ew != ew else f"{ew:>7.1%}"
        print(
            f"  {it:>4} {c_str:>7} {p:>9.4f} {w:>4d}/{N_POP} "
            f"{ew_str} {d_str:>8} {g:>10.2f}"
        )

    eval_iters = list(history.get("iteration", []))
    eval_succs = list(history.get("eval_success", []))
    print("\n=== Per-iter eval rollouts (greedy, on full 16-word pool) ===")
    for it, s in zip(eval_iters, eval_succs):
        print(f"  iter={it:>4}  eval_success={s:.1%}")

    post_ws_indist = float(
        history.get("post_warmstart_success_indist", [float("nan")])[0]
    )
    end_es_greedy_history = float(eval_succs[-1]) if eval_succs else float("nan")

    ws_gain_greedy = post_ws_indist - pre_ws_greedy
    es_gain_greedy = end_es_greedy_history - post_ws_indist

    # Stochastic version: we lack a post-WS stochastic measurement mid-pipeline,
    # so we compare pre-WS stochastic to final stochastic and attribute the
    # delta to WS+ES jointly. For "ES credit" purposes this is a ceiling, not
    # a perfect decomposition -- see the greedy numbers for the strict test.
    joint_gain_stoch = end_stoch - pre_ws_stoch

    nonzero_dprobe = [d for d in train_dprobe if d == d and abs(d) > 1e-9]
    eval_dprobe_total = sum(1 for d in train_dprobe if d == d)
    nonzero_frac = (
        len(nonzero_dprobe) / max(1, eval_dprobe_total) if eval_dprobe_total else 0.0
    )

    print("\n=== EXPERIMENT 2 SUMMARY ===")
    print("  [GREEDY, full 16-word pool, deterministic argmax]")
    print(f"    pre-warm-start success      : {pre_ws_greedy:.1%}")
    print(
        f"    post-warm-start success     : {post_ws_indist:.1%}   "
        f"(ws_gain = {ws_gain_greedy:+.1%})"
    )
    print(
        f"    post-ES success (final)     : {end_es_greedy_history:.1%}   "
        f"(es_gain = {es_gain_greedy:+.1%})"
    )
    if eval_succs:
        peak_es = max(eval_succs)
        print(f"    peak eval_success (any iter): {peak_es:.1%}")

    print("  [STOCHASTIC, full 16-word pool, temp-1 sampling]")
    print(f"    pre-warm-start success      : {pre_ws_stoch:.1%}")
    print(
        f"    final (post-WS+ES) success  : {end_stoch:.1%}   "
        f"(joint_gain = {joint_gain_stoch:+.1%})"
    )

    print(
        f"  dprobe non-zero fraction    : {len(nonzero_dprobe)}/{eval_dprobe_total} = "
        f"{nonzero_frac:.0%}"
    )

    pass_es_gain_greedy_10 = (es_gain_greedy == es_gain_greedy) and (
        es_gain_greedy >= 0.10
    )
    pass_dprobe = nonzero_frac >= 0.25

    print(
        f"\n  PASS criteria (experiment brief):\n"
        f"    greedy es_gain >= +10pp  -> {pass_es_gain_greedy_10}\n"
        f"    dprobe non-zero >=25%    -> {pass_dprobe}"
    )

    if pass_es_gain_greedy_10 and pass_dprobe:
        print(
            "\n  VERDICT: PASS. Mini-batch ES under CRN recovered signal at vocab=16; "
            "the week-12 bottleneck-fix hypothesis is validated."
        )
    else:
        print(
            "\n  VERDICT: FAIL (brief criteria). Review the per-iter table + stochastic "
            "numbers. If stochastic climbed but greedy didn't, it's a metric-saturation "
            "issue like Exp 1. If both are flat but dprobe non-zero fires, calibration "
            "or step-size may be the next axis to tune. If dprobe is zero-dominated, "
            "the subset-CRN plumbing is suspect -- re-verify."
        )


if __name__ == "__main__":
    main()
