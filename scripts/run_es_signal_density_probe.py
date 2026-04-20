#!/usr/bin/env python3
"""
Run the ES-only signal probe with PROBE_VOCAB=4 (signal-density / Test B).

Mirrors `notebooks/week12_implementation_LoRARun.ipynb` probe cell so you can
execute without Jupyter. From repo root:

    python3 scripts/run_es_signal_density_probe.py

Optional: PROBE_ITERS=10 python3 scripts/run_es_signal_density_probe.py
(use the same interpreter / venv as your Jupyter kernel if system python lacks torch)
"""
from __future__ import annotations

import gc
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordle_env import load_wordle_environment  # noqa: E402
from wordle_gpt2_policy import WordleGPT2Policy, TRANSFORMERS_AVAILABLE  # noqa: E402

import es_wordle  # noqa: E402

importlib.reload(es_wordle)
from es_wordle import es_gradient_estimate_wordle, train_es_wordle  # noqa: E402

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
            "Chat-template models require jinja2>=3.1.0. "
            "Install with: python -m pip install -U 'jinja2>=3.1.0'"
        ) from exc
    installed = getattr(jinja2, "__version__", "0")
    if _parse_version_tuple(installed) < (3, 1, 0):
        raise ImportError(
            f"Chat-template models require jinja2>=3.1.0, found {installed}."
        )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_model_load_kwargs(device: torch.device) -> tuple[dict, str]:
    if device.type != "cuda":
        return {}, "float32"
    if torch.cuda.is_bf16_supported():
        return {"torch_dtype": torch.bfloat16}, "bfloat16"
    return {"torch_dtype": torch.float16}, "float16"


def main() -> None:
    # --- Match notebook cell 4 (gemma_full) ---
    SEED = 42
    DEVICE = choose_device()
    MOCK_ENV = False
    USE_PRIME_TARGETS = True
    USE_LORA = True
    LORA_R = 8
    RICHER_PROMPT = True
    SIGMA = 0.02
    ALPHA = None  # noqa: F841  # probe auto-calibrates
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
    MAX_VOCAB = 1024
    NUM_TRAIN_EXAMPLES = 2000
    NUM_EVAL_EXAMPLES = 20
    N_EVAL_EPISODES = 16
    EVAL_N_EPISODES = 50
    EVAL_EVERY = 1

    require_chat_template_support()
    MODEL_LOAD_KWARGS, dtype_name = default_model_load_kwargs(DEVICE)

    PROBE_VOCAB = 4
    PROBE_N_POP = 64
    PROBE_N_EVAL = 16
    PROBE_ITERS = int(os.environ.get("PROBE_ITERS", "20"))
    PROBE_TARGET_STEP = 0.13
    _ALPHA_BASE_FALLBACK = ALPHA if ALPHA is not None else 1.26e-05

    print(f"ROOT: {ROOT}")
    print(f"device={DEVICE} dtype={dtype_name} model={MODEL_NAME} LoRA r={LORA_R}")
    print(
        f"Signal-density probe: PROBE_VOCAB={PROBE_VOCAB} N_POP={PROBE_N_POP} "
        f"n_eval_episodes={PROBE_N_EVAL} iters={PROBE_ITERS} "
        f"baseline_subtract={BASELINE_SUBTRACT} ema_beta={EMA_BETA}"
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = DEVICE
    probe_policy = WordleGPT2Policy(
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

    probe_env = load_wordle_environment(
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_eval_examples=NUM_EVAL_EXAMPLES,
        use_prime_intellect=not MOCK_ENV,
        target_pool=probe_policy.words,
    )
    probe_env.set_target_pool(list(probe_policy.words[:PROBE_VOCAB]))

    _probe_state_py = random.getstate()
    _probe_state_np = np.random.get_state()
    _probe_state_torch = torch.get_rng_state()
    _probe_state_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        _g, _af, _fits, _aw, _ps, _wrs = es_gradient_estimate_wordle(
            probe_policy,
            probe_env,
            N=PROBE_N_POP,
            sigma=SIGMA,
            n_eval_episodes=PROBE_N_EVAL,
            max_turns=6,
            rank_fitness=RANK_FITNESS,
            fitness_objective=FITNESS_OBJECTIVE,
            win_fitness_scale=WIN_FITNESS_SCALE,
            antithetic=ANTITHETIC,
            common_random_numbers=COMMON_RANDOM_NUMBERS,
            baseline_subtract=BASELINE_SUBTRACT,
        )
        _g_norm_probe = float(_g.norm().item())
        if _g_norm_probe > 1e-8:
            PROBE_ALPHA = float(PROBE_TARGET_STEP / _g_norm_probe)
            print(
                f"  probe ALPHA auto-cal: ‖ĝ‖={_g_norm_probe:.4g} -> "
                f"PROBE_ALPHA={PROBE_ALPHA:.2e} (target step ≈ {PROBE_TARGET_STEP})"
            )
        else:
            PROBE_ALPHA = 4.0 * _ALPHA_BASE_FALLBACK
            print(
                f"  probe ALPHA fallback: ‖ĝ‖≈0. Using PROBE_ALPHA={PROBE_ALPHA:.2e}"
            )
    finally:
        random.setstate(_probe_state_py)
        np.random.set_state(_probe_state_np)
        torch.set_rng_state(_probe_state_torch)
        if _probe_state_cuda is not None:
            torch.cuda.set_rng_state_all(_probe_state_cuda)

    probe_history = train_es_wordle(
        probe_policy,
        probe_env,
        N=PROBE_N_POP,
        sigma=SIGMA,
        alpha=PROBE_ALPHA,
        n_iterations=PROBE_ITERS,
        n_eval_episodes=PROBE_N_EVAL,
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

    _grad_cos = [
        float(c) for c in probe_history.get("train_grad_cos", []) if c == c
    ]
    _late_cos = _grad_cos[5:] if len(_grad_cos) >= 6 else _grad_cos
    _med_cos = float(np.median(_late_cos)) if _late_cos else float("nan")
    _evals = list(probe_history.get("eval_success", []))
    _eval_lift = (
        float(_evals[-1]) - float(_evals[0]) if len(_evals) >= 2 else float("nan")
    )
    _pass_cos = _med_cos == _med_cos and _med_cos > 0.05
    _pass_lift = _eval_lift == _eval_lift and _eval_lift >= 0.15

    print("\n=== Probe verdict ===")
    print(
        f"  median cos(ĝ), iters >=5: {_med_cos:+.3f}   "
        f"(pass: > 0.05 -> {_pass_cos})"
    )
    print(
        f"  eval_success lift:        {_eval_lift:+.1%}   "
        f"(pass: >= +15pp -> {_pass_lift})"
    )
    if _pass_cos and _pass_lift:
        print("  PASS")
    else:
        print("  FAIL (signal still variance-dominated or no sustained eval lift)")

    del probe_policy, probe_env, probe_history
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
