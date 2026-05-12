#!/usr/bin/env python3
"""
Headless runner for ``notebooks/week16_impl.ipynb`` ES training (Qwen + LoRA).

Survives SSH disconnects when launched inside ``tmux``/``screen`` (or ``nohup``),
unlike a Jupyter kernel that may be tied to the IDE.

Usage (repo root, log everything)::

    .venv/bin/python -u scripts/run_week16_es.py 2>&1 | tee logs/week16_es.log

Skip the expensive ALPHA probe (set learning rate explicitly)::

    .venv/bin/python -u scripts/run_week16_es.py --skip-alpha-probe --alpha 0.09 2>&1 | tee logs/week16_es.log

Optional: ``bash scripts/run_week16_es_tmux.sh`` starts a tmux session and tees to
``logs/week16_es_<timestamp>.log`` automatically.

Survives disconnects (no Slurm required)::

    bash scripts/run_week16_es_nohup.sh              # nohup + auto artifacts dir
    bash scripts/run_week16_es_tmux.sh               # tmux + tee
    WEEK16_DETACH_MODE=nohup bash scripts/run_week16_es_detached.sh

Slurm (only on real clusters)::

    sbatch scripts/slurm/week16_es.sbatch
"""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_mpl_conf = ROOT / ".cache" / "matplotlib"
try:
    _mpl_conf.mkdir(parents=True, exist_ok=True)
except OSError:
    _mpl_conf = Path(tempfile.gettempdir()) / "matplotlib-week16-es"
    _mpl_conf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_conf.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wordle_env import load_wordle_environment  # noqa: E402
import wordle_qwen_policy  # noqa: E402
importlib.reload(wordle_qwen_policy)
from wordle_qwen_policy import WordleQwenPolicy  # noqa: E402

import es_wordle  # noqa: E402
importlib.reload(es_wordle)
from es_wordle import es_gradient_estimate_wordle, train_es_wordle  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_dtype_kwargs() -> dict:
    if not torch.cuda.is_available():
        return {}
    if torch.cuda.is_bf16_supported():
        return {"torch_dtype": torch.bfloat16}
    return {"torch_dtype": torch.float16}


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Week16 Wordle ES (Qwen LoRA), headless.")
    parser.add_argument(
        "--skip-alpha-probe",
        action="store_true",
        help="Skip es_gradient_estimate_wordle probe (fast start). Requires --alpha.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        metavar="FLOAT",
        help="ES step size. Required if --skip-alpha-probe. Else None => probe sets it.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving the matplotlib figure.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Where to write PNG (overrides default under --artifacts-dir if set).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Write plots/, lora_adapter/, es_history.pkl under this directory (Slurm-friendly).",
    )
    args = parser.parse_args()

    if args.skip_alpha_probe and args.alpha is None:
        parser.error("--skip-alpha-probe requires --alpha")

    if args.artifacts_dir is not None:
        artifacts_base = args.artifacts_dir.expanduser().resolve()
        artifacts_base.mkdir(parents=True, exist_ok=True)
        plots_dir = artifacts_base / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        default_plot_path = plots_dir / "training_curves.png"
        adapter_dir = artifacts_base / "lora_adapter"
        history_path = artifacts_base / "es_history.pkl"
    else:
        artifacts_base = None
        default_plot_path = ROOT / "models" / "week16_es_training_curves.png"
        adapter_dir = ROOT / "models" / "wordle_qwen_es_lora.week16"
        history_path = ROOT / "models" / "wordle_qwen_es_history.week16.pkl"

    plot_path_resolved: Path = args.plot_path.expanduser().resolve() if args.plot_path else default_plot_path

    # === Hyperparameters (mirror notebook §2) =================================
    SEED = 42
    # ES optimizes LoRA on top of this base checkpoint (RL fine-tune, not SFT).
    MODEL_SFT = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"  # reference / ceiling comparisons
    MODEL_RL_REF = "PrimeIntellect/Qwen3-1.7B-Wordle-RL"
    MODEL_ES_BASE = MODEL_RL_REF

    MAX_PROMPT_LENGTH = 1024
    ENABLE_THINKING = True
    MAX_NEW_TOKENS = 512
    GEN_TEMPERATURE = 0.8
    GEN_TOP_P = 0.9

    LORA_R = 4
    LORA_ALPHA = 16
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    CAST_LORA_TO_FP32 = True

    N_POP = 8
    N_ITER = 15
    N_EVAL_EPISODES = 8
    EVAL_N_EPISODES = 16
    EVAL_EVERY = 1
    PROBE_N_EPISODES = 8
    MAX_TURNS = 6
    SIGMA = 0.02
    ALPHA: float | None = args.alpha if args.alpha is not None else None
    # Raw ES applies θ ← θ + α·ĝ; ‖ĝ‖ on ~1–2M LoRA dims spikes across iters → ‖α·ĝ‖ explosions.
    # Normalized θ ← θ + α·ĝ/‖ĝ‖ fixes step size ‖Δθ‖=α each iter (ALPHA probe sets α ≈ step target).
    NORMALIZE_GRADIENT = True
    RANK_FITNESS = True
    BASELINE_SUBTRACT = True
    ANTITHETIC = True
    COMMON_RANDOM_NUMBERS = True
    EMA_BETA = 0.0
    EVAL_DETERMINISTIC = True
    FITNESS_OBJECTIVE = "win_plus_return"
    WIN_FITNESS_SCALE = 8.0
    PER_ITER_SECRET_SUBSET_SIZE = 8
    TRACK_BEST_ITER = True
    # Last iterate can be worse than an earlier peak (see prior run: best 12.5% vs final 0%).
    RESTORE_BEST_ON_FINISH = True

    RL_ceiling = float("nan")
    SFT_cold = float("nan")

    MOCK_ENV = False
    NUM_TRAIN_EXAMPLES = 2000
    NUM_EVAL_EXAMPLES = 100

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = choose_device()
    MODEL_LOAD_KWARGS = cuda_dtype_kwargs()

    _log(f"[week16_es] ROOT={ROOT}")
    if artifacts_base is not None:
        _log(f"[week16_es] artifacts_dir={artifacts_base}")
    _log(f"[week16_es] device={DEVICE} model_load_kwargs={MODEL_LOAD_KWARGS or 'float32'}")
    _log(f"[week16_es] MODEL_ES_BASE={MODEL_ES_BASE}")
    if torch.cuda.is_available():
        _log(f"[week16_es] cuda_device={torch.cuda.get_device_name(0)}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    policy = WordleQwenPolicy(
        model_name=MODEL_ES_BASE,
        max_prompt_length=MAX_PROMPT_LENGTH,
        enable_thinking=ENABLE_THINKING,
        max_new_tokens=MAX_NEW_TOKENS,
        gen_temperature=GEN_TEMPERATURE,
        gen_top_p=GEN_TOP_P,
        use_lora=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_target_modules=LORA_TARGET_MODULES,
        cast_lora_to_fp32=CAST_LORA_TO_FP32,
        model_kwargs=MODEL_LOAD_KWARGS,
    ).to(DEVICE)

    _log(
        f"[week16_es] trainable={policy.count_trainable_parameters():,} "
        f"total={policy.count_parameters():,} "
        f"action_granularity={policy.action_granularity}"
    )
    _trainable = policy.count_trainable_parameters()
    if _trainable > 50_000_000:
        _log(f"[week16_es] WARN trainable={_trainable:,} (expected <2M LoRA)")

    env_train = load_wordle_environment(
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_eval_examples=NUM_EVAL_EXAMPLES,
        use_prime_intellect=not MOCK_ENV,
    )
    env_eval = load_wordle_environment(
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_eval_examples=NUM_EVAL_EXAMPLES,
        use_prime_intellect=not MOCK_ENV,
    )
    _log("[week16_es] envs ready (train + eval)")

    # === ALPHA probe (optional) ==============================================
    if not args.skip_alpha_probe:
        _target_step = 0.13
        _min_step = 0.05

        _py_state = random.getstate()
        _np_state = np.random.get_state()
        _torch_state = torch.get_rng_state()
        _cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            _grad, _avg_fit, _fits, _avg_win, _pop_std, _es_wrs = es_gradient_estimate_wordle(
                policy,
                env_train,
                N=N_POP,
                sigma=SIGMA,
                n_eval_episodes=N_EVAL_EPISODES,
                max_turns=MAX_TURNS,
                rank_fitness=RANK_FITNESS,
                fitness_objective=FITNESS_OBJECTIVE,
                win_fitness_scale=WIN_FITNESS_SCALE,
                antithetic=ANTITHETIC,
                common_random_numbers=COMMON_RANDOM_NUMBERS,
                baseline_subtract=BASELINE_SUBTRACT,
                per_iter_secret_subset_size=PER_ITER_SECRET_SUBSET_SIZE,
            )
            _g_norm = float(_grad.norm().item())
            _ess_rank = len({round(float(f), 6) for f in _fits})
            _win_count = sum(1 for wr in _es_wrs if wr > 0.0)
            if _g_norm <= 1e-8:
                raise RuntimeError(
                    "ALPHA probe ‖ĝ‖ ≈ 0 — check generation/parsing or try --skip-alpha-probe --alpha ..."
                )
            if NORMALIZE_GRADIENT:
                # Update is α·ĝ/‖ĝ‖ ⇒ per-iter ‖Δθ‖ = α; ‖ĝ‖ is diagnostic only here.
                _suggested_alpha = float(_target_step)
            else:
                _suggested_alpha = float(_target_step / _g_norm)
            if ALPHA is None:
                ALPHA = _suggested_alpha
                _action = f"  AUTO ALPHA = {ALPHA:.2e}"
            else:
                _action = (
                    f"  manual ALPHA = {ALPHA:.2e} "
                    f"(calibrator suggested {_suggested_alpha:.2e})"
                )
            if NORMALIZE_GRADIENT:
                _implied_step = float(ALPHA)
            else:
                _implied_step = ALPHA * _g_norm
            _step_line = (
                f"  ‖Δθ‖ per ES step = {_implied_step:.4g}   (normalized updates; "
                f"target ≈ {_target_step})\n"
                if NORMALIZE_GRADIENT
                else (
                    f"  ALPHA * ‖ĝ‖       = {_implied_step:.4g}   (target ≈ {_target_step})\n"
                )
            )
            _log(
                f"\n[ALPHA calibration @ init, trainable={policy.count_trainable_parameters():,}]\n"
                f"  raw ‖ĝ‖           = {_g_norm:.4g}\n"
                f"{_step_line}"
                f"  ES probe avg_fit  = {_avg_fit:+.3f}   ES_win = {_avg_win:.1%}   popσ = {_pop_std:.4f}\n"
                f"  ES signal probes  = ess_rank {_ess_rank}/{N_POP}, wins {_win_count}/{N_POP}\n"
                f"{_action}"
            )
            if _implied_step < _min_step:
                _log(
                    f"  WARNING implied step {_implied_step:.4g} < {_min_step}. Suggested ALPHA ≈ {_suggested_alpha:.2e}"
                )
        finally:
            random.setstate(_py_state)
            np.random.set_state(_np_state)
            torch.set_rng_state(_torch_state)
            if _cuda_state is not None:
                torch.cuda.set_rng_state_all(_cuda_state)
            free_gpu()
    else:
        assert ALPHA is not None
        _log(f"[week16_es] skipped ALPHA probe; using ALPHA={ALPHA:.2e}")
        if NORMALIZE_GRADIENT:
            _log(
                "[week16_es] hint: with normalize_gradient=True, ‖Δθ‖=ALPHA each iter "
                "(typical range ~5e-2–2e-1)."
            )

    assert ALPHA is not None
    _log(
        f"[week16_es] NORMALIZE_GRADIENT={NORMALIZE_GRADIENT}  EMA_BETA={EMA_BETA}  "
        f"N_ITER={N_ITER}  SIGMA={SIGMA}  train_ep/perturb={N_EVAL_EPISODES}  "
        f"eval_ep={EVAL_N_EPISODES}"
    )

    _log("[week16_es] starting train_es_wordle ...")
    history = train_es_wordle(
        policy,
        env_train,
        N=N_POP,
        sigma=SIGMA,
        alpha=ALPHA,
        n_iterations=N_ITER,
        n_eval_episodes=N_EVAL_EPISODES,
        max_turns=MAX_TURNS,
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
        env_eval=env_eval,
        probe_n_episodes=PROBE_N_EPISODES,
        baseline_subtract=BASELINE_SUBTRACT,
        per_iter_secret_subset_size=PER_ITER_SECRET_SUBSET_SIZE,
        track_best_iter=TRACK_BEST_ITER,
        restore_best_on_finish=RESTORE_BEST_ON_FINISH,
    )

    final_eval_success = (
        float(history["eval_success"][-1]) if history.get("eval_success") else float("nan")
    )
    best_raw = history.get("best_eval_success")
    best_eval_success = float(best_raw[0]) if best_raw else float("nan")
    best_iter = int(history["best_iter"][0]) if history.get("best_iter") else -1
    # With restore_best_on_finish, saved weights = best checkpoint; last-iter eval is high-variance
    # (EVAL_N_EPISODES greedy games), not the headline for the adapter on disk.
    exported_matches_best = bool(
        RESTORE_BEST_ON_FINISH
        and np.isfinite(best_eval_success)
        and best_iter >= 0
    )
    _exported_eval = best_eval_success if exported_matches_best else final_eval_success

    _lines = [
        f"\n=== ES finished ===\n",
        (
            "  NOTE: Exported LoRA = best-so-far checkpoint when restore_best_on_finish=True; "
            "last checkpoint % is greedy eval variance (EVAL_N_EPISODES games).\n"
            if exported_matches_best
            else ""
        ),
        f"  last checkpoint eval_success (iter {N_ITER - 1}) : {final_eval_success:.1%}\n",
        f"  best greedy eval_seen (iter {best_iter})        : {best_eval_success:.1%}\n",
    ]
    if np.isfinite(SFT_cold):
        _lines.append(
            f"  ES contribution (exported − SFT_cold) : {(_exported_eval - SFT_cold):+.1%}\n"
        )
    if np.isfinite(RL_ceiling):
        _lines.append(
            f"  Gap to RL ceiling (ceiling − exported) : {(RL_ceiling - _exported_eval):+.1%}\n"
        )
    _log("".join(_lines))

    # === Plots ===============================================================
    if not args.no_plots:
        plot_path_resolved.parent.mkdir(parents=True, exist_ok=True)

        it = np.array(history.get("iteration", []), dtype=float)
        ti = np.array(history.get("train_iter", []), dtype=float)

        fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=False)

        axes[0, 0].plot(
            it, history.get("eval_success", []), "g-o", ms=3, label="eval success"
        )
        if np.isfinite(SFT_cold):
            axes[0, 0].axhline(
                SFT_cold, color="gray", ls=":", lw=0.9, label=f"SFT_cold={SFT_cold:.1%}"
            )
        if np.isfinite(RL_ceiling):
            axes[0, 0].axhline(
                RL_ceiling, color="red", ls=":", lw=0.9, label=f"RL_ceiling={RL_ceiling:.1%}"
            )
        axes[0, 0].set_title("Eval success rate")
        axes[0, 0].set_xlabel("iteration")
        axes[0, 0].set_ylim(0, 1.05)
        _leg_h, leg_l = axes[0, 0].get_legend_handles_labels()
        if any(leg_l):
            axes[0, 0].legend(loc="lower right", fontsize=7)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(it, history.get("eval_reward", []), "b-o", ms=3)
        axes[0, 1].set_title("Eval mean reward")
        axes[0, 1].set_xlabel("iteration")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(it, history.get("eval_turns", []), "m-o", ms=3)
        axes[0, 2].set_title("Eval mean turns")
        axes[0, 2].set_xlabel("iteration")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(ti, history.get("train_fitness", []), color="c", lw=1)
        axes[1, 0].set_title("ES mean fitness (per iter)")
        axes[1, 0].set_xlabel("iteration")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(ti, history.get("train_grad_norm", []), color="r", lw=1)
        axes[1, 1].set_title("Applied gradient norm")
        axes[1, 1].set_xlabel("iteration")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(ti, history.get("param_drift", []), color="k", lw=1)
        axes[1, 2].set_title("Parameter drift ‖θ − θ₀‖")
        axes[1, 2].set_xlabel("iteration")
        axes[1, 2].grid(True, alpha=0.3)

        axes[2, 0].plot(ti, history.get("pop_fitness_std", []), color="orange", lw=1)
        axes[2, 0].set_title("Population fitness std")
        axes[2, 0].set_xlabel("iteration")
        axes[2, 0].grid(True, alpha=0.3)

        gc_arr = np.array(history.get("train_grad_cos", []), dtype=float)
        axes[2, 1].plot(ti, gc_arr, color="purple", lw=1)
        axes[2, 1].axhline(0.0, color="gray", lw=0.7, ls=":")
        axes[2, 1].set_title("Gradient cosine cos(g_t, g_{t-1})")
        axes[2, 1].set_xlabel("iteration")
        axes[2, 1].set_ylim(-1.05, 1.05)
        axes[2, 1].grid(True, alpha=0.3)

        ess_arr = np.array(history.get("train_ess_rank", []), dtype=float)
        axes[2, 2].plot(ti, ess_arr, color="teal", lw=1)
        axes[2, 2].axhline(N_POP, color="gray", lw=0.7, ls=":", label=f"N_POP={N_POP}")
        axes[2, 2].set_title("ESS rank (unique fitness count)")
        axes[2, 2].set_xlabel("iteration")
        axes[2, 2].set_ylim(0, max(2, N_POP + 1))
        axes[2, 2].legend(loc="lower right", fontsize=7)
        axes[2, 2].grid(True, alpha=0.3)

        probe_delta = np.array(history.get("train_probe_delta", []), dtype=float)
        finite_probe = np.isfinite(probe_delta)
        axes[3, 0].axhline(0.0, color="gray", lw=0.7, ls=":")
        axes[3, 0].plot(ti[finite_probe], probe_delta[finite_probe], "o-", ms=3, color="firebrick", lw=1)
        axes[3, 0].set_title("Δ probe success (post − pre)")
        axes[3, 0].set_xlabel("iteration")
        axes[3, 0].grid(True, alpha=0.3)

        fb_rate = np.array(history.get("train_trie_fallback_rate", []), dtype=float)
        axes[3, 1].plot(ti, fb_rate, color="brown", lw=1)
        axes[3, 1].set_title("Parse-failure rate (gen-mode)")
        axes[3, 1].set_xlabel("iteration")
        axes[3, 1].set_ylim(0, 1.05)
        axes[3, 1].grid(True, alpha=0.3)

        parse_attempts = np.array(history.get("train_trie_steps", []), dtype=float)
        oov = np.array(history.get("train_trie_oov_words", []), dtype=float)
        axes[3, 2].plot(ti, parse_attempts, color="slateblue", lw=1, label="parse attempts")
        axes[3, 2].plot(ti, oov, color="darkred", lw=1, label="oov words")
        axes[3, 2].set_title("Generation volume")
        axes[3, 2].set_xlabel("iteration")
        axes[3, 2].legend(loc="upper right", fontsize=7)
        axes[3, 2].grid(True, alpha=0.3)

        plt.suptitle(
            f"Wordle ES on RL base | {MODEL_ES_BASE} | LoRA r={LORA_R} | thinking={ENABLE_THINKING}"
        )
        plt.tight_layout()
        fig.savefig(plot_path_resolved, dpi=150)
        plt.close(fig)
        _log(f"[week16_es] saved plot: {plot_path_resolved}")

    # === Summary + save (mirror notebook §5) ================================
    es_contribution = (
        _exported_eval - SFT_cold if np.isfinite(SFT_cold) else float("nan")
    )
    gap_to_ceiling = (
        RL_ceiling - _exported_eval if np.isfinite(RL_ceiling) else float("nan")
    )

    _log("=" * 60)
    _log("FINAL ATTRIBUTION")
    _log("=" * 60)
    _rl = f"{RL_ceiling:>6.1%}" if np.isfinite(RL_ceiling) else "   n/a"
    _sft = f"{SFT_cold:>6.1%}" if np.isfinite(SFT_cold) else "   n/a"
    _log(f"  RL_ceiling (their full pipeline) : {_rl}")
    _log(f"  SFT_cold   (start of ES)         : {_sft}")
    _log(
        f"  SFT + ES   (exported adapter %)   : {_exported_eval:>6.1%}"
        + ("  (= best_seen; restored before save)" if exported_matches_best else "")
    )
    _log(
        f"  SFT + ES   (best_seen @ iter {best_iter:>3d})     : {best_eval_success:>6.1%}"
    )
    _log(f"  SFT + ES   (last checkpoint %)   : {final_eval_success:>6.1%}")
    _log("-" * 60)
    if np.isfinite(es_contribution):
        _log(
            f"  ES contribution                  : {es_contribution:+6.1%}  (exported − SFT_cold)"
        )
    else:
        _log("  ES contribution                  : n/a")
    if np.isfinite(gap_to_ceiling):
        _log(
            f"  Gap to ceiling                   : {gap_to_ceiling:+6.1%}  (RL_ceiling − exported)"
        )
    else:
        _log("  Gap to ceiling                   : n/a")
    _log("=" * 60)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    try:
        policy.lm.save_pretrained(str(adapter_dir))
        _log(f"[week16_es] saved LoRA adapter: {adapter_dir}")
    except Exception as exc:  # noqa: BLE001
        _log(f"[week16_es] adapter save skipped ({exc.__class__.__name__}: {exc})")

    try:
        history_save = {k: v for k, v in history.items() if k != "best_params"}
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("wb") as f:
            pickle.dump(
                {
                    "history": history_save,
                    "RL_ceiling": RL_ceiling,
                    "SFT_cold": SFT_cold,
                    "final_eval_success": final_eval_success,
                    "exported_eval_success": _exported_eval,
                    "restore_best_weights_match_best_seen": exported_matches_best,
                    "best_eval_success": best_eval_success,
                    "best_iter": best_iter,
                    "hparams": {
                        "MODEL_SFT": MODEL_SFT,
                        "MODEL_RL_REF": MODEL_RL_REF,
                        "MODEL_ES_BASE": MODEL_ES_BASE,
                        "LORA_R": LORA_R,
                        "N_POP": N_POP,
                        "N_ITER": N_ITER,
                        "N_EVAL_EPISODES": N_EVAL_EPISODES,
                        "EVAL_N_EPISODES": EVAL_N_EPISODES,
                        "SIGMA": SIGMA,
                        "ALPHA": ALPHA,
                        "EMA_BETA": EMA_BETA,
                        "ENABLE_THINKING": ENABLE_THINKING,
                        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
                        "PER_ITER_SECRET_SUBSET_SIZE": PER_ITER_SECRET_SUBSET_SIZE,
                        "NORMALIZE_GRADIENT": NORMALIZE_GRADIENT,
                    },
                },
                f,
            )
        _log(f"[week16_es] saved history: {history_path}")
    except Exception as exc:  # noqa: BLE001
        _log(f"[week16_es] history pickle skipped ({exc.__class__.__name__}: {exc})")

    _log("[week16_es] done.")


if __name__ == "__main__":
    main()
