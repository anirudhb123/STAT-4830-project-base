"""
Evolution Strategies training for Wordle environment.

Adapted from the GridWorld ES implementation to work with Wordle.
"""

import random
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Sequence, Callable, Any, Union


def _snapshot_rng_state(device: torch.device) -> dict:
    """Capture RNG state for Python/numpy/torch (+ CUDA if available).

    Used for common-random-numbers (CRN) in ES: every population member in the
    same iteration is evaluated against an identical stream of env + policy
    randomness, which dramatically reduces the variance of the ES rank
    ordering at small N.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _trainable_params(policy):
    """Parameters ES should optimize (excludes frozen backbone weights)."""
    return [p for p in policy.parameters() if p.requires_grad]


def _set_flat_params(policy, flat_params: torch.Tensor):
    """Set trainable policy parameters from a flattened parameter vector."""
    offset = 0
    for param in _trainable_params(policy):
        param_length = param.numel()
        param_slice = flat_params[offset:offset + param_length]
        param.data = param_slice.to(device=param.device, dtype=param.dtype).view_as(param).clone()
        offset += param_length


def _evaluate_perturbation_serial(policy, env, n_eval_episodes, max_turns):
    """Original one-game-at-a-time fallback (used for policies without forward_logits_batch)."""
    fitness = 0.0
    wins = 0
    for _ in range(n_eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        turns = 0
        info: dict = {}

        policy.eval()
        with torch.no_grad():
            while not done and turns < max_turns:
                state_embedding = env.get_state_embedding(state)

                if hasattr(policy, "format_action_xml"):
                    action_xml, _ = policy.format_action_xml(
                        state, state_embedding, deterministic=False
                    )
                else:
                    action_idx, _ = policy.get_action(state_embedding, deterministic=False)
                    word = policy.vocab.action_to_word(action_idx)
                    action_xml = f"<guess>{word}</guess>"

                state, reward, done, info = env.step(action_xml)
                episode_reward += reward
                turns += 1

        fitness += episode_reward
        if float(info.get("correct_answer", 0.0)) >= 0.5:
            wins += 1

    mean_ret = fitness / n_eval_episodes
    win_rate = wins / max(1, n_eval_episodes)
    return mean_ret, win_rate


def _format_action_xml_for_word(state, word: str) -> str:
    """Replicate WordleGPT2Policy.format_action_xml's wrapper without re-running the LM."""
    turn = state.turn_number + 1
    if turn == 1:
        think = f"Using the language model prior, opening with {word}."
    else:
        think = f"Conditioning on feedback, next guess: {word}."
    return f"<think>{think}</think>\n<guess>{word}</guess>"


def _rollout_batched(
    policy,
    env,
    n_episodes: int,
    max_turns: int,
    deterministic: bool = False,
) -> Tuple[List[float], List[float], List[int]]:
    """Play ``n_episodes`` games in lockstep, batching the LM forward across active games.

    The single env instance is shared: ``env.current_state`` is swapped to each
    game's state immediately before that game's ``env.step`` call, so the env's
    internal accounting still works. Action selection matches
    ``WordleGPT2Policy.get_action`` semantics: previous-guesses are masked to
    -inf, then ``argmax`` (``deterministic=True``) or a single ``Categorical``
    sample on the softmax (``deterministic=False``).

    Returns three per-episode lists of length ``n_episodes`` (in order, no
    aggregation): ``rewards``, ``successes`` (1.0/0.0 from
    ``info["correct_answer"]``), ``turn_counts``.
    """
    states = [env.reset() for _ in range(n_episodes)]
    rewards = [0.0] * n_episodes
    done_flags = [False] * n_episodes
    infos: List[Dict[str, Any]] = [{} for _ in range(n_episodes)]
    turn_counts = [0] * n_episodes

    policy.eval()
    with torch.no_grad():
        for _t in range(max_turns):
            active = [i for i, d in enumerate(done_flags) if not d]
            if not active:
                break

            active_states = [states[i] for i in active]
            logits_b = policy.forward_logits_batch(active_states)  # [len(active), action_dim]

            # Mask previous-guess actions to -inf per row (matches
            # WordleGPT2Policy.get_action behavior).
            word_to_idx = getattr(policy, "word_to_idx", None)
            if word_to_idx is not None:
                logits_b = logits_b.clone()
                for j, i in enumerate(active):
                    for g in states[i].previous_guesses:
                        u = g.upper()
                        idx = word_to_idx.get(u)
                        if idx is not None:
                            logits_b[j, idx] = float("-inf")

            if deterministic:
                action_idxs = torch.argmax(logits_b, dim=-1)
            else:
                probs = torch.softmax(logits_b, dim=-1)
                action_idxs = torch.distributions.Categorical(probs=probs).sample()

            idx_to_word = getattr(policy, "idx_to_word", None)
            for j, i in enumerate(active):
                aidx = int(action_idxs[j].item())
                if idx_to_word is not None:
                    word = idx_to_word[aidx]
                else:
                    word = policy.vocab.action_to_word(aidx)
                action_xml = _format_action_xml_for_word(states[i], word)

                env.current_state = states[i]
                next_state, r, d, info = env.step(action_xml)
                states[i] = next_state
                rewards[i] += float(r)
                done_flags[i] = bool(d)
                infos[i] = info
                turn_counts[i] += 1

    successes = [float(info.get("correct_answer", 0.0)) for info in infos]
    return rewards, successes, turn_counts


def _evaluate_perturbation_batched(policy, env, n_eval_episodes, max_turns):
    """ES fitness rollout: stochastic actions, returns aggregated (mean_return, win_rate)."""
    rewards, successes, _turns = _rollout_batched(
        policy, env, n_eval_episodes, max_turns, deterministic=False
    )
    fitness = float(sum(rewards))
    wins = sum(1 for s in successes if s >= 0.5)
    mean_ret = fitness / max(1, n_eval_episodes)
    win_rate = wins / max(1, n_eval_episodes)
    return mean_ret, win_rate


def _evaluate_perturbation(policy, env, perturbed_params, n_eval_episodes, max_turns):
    """
    Evaluate fitness of a perturbed policy.

    Returns (mean_episode_return, win_rate) where win_rate is the fraction of
    eval episodes that ended with a correct guess. Wordle rewards mix partial
    credit into the return, so mean_return can stay ~flat even when win_rate is 0%.

    Uses ``policy.forward_logits_batch`` to batch the LM forward across the
    ``n_eval_episodes`` parallel games when available; falls back to the
    one-game-at-a-time loop for policies that don't expose it (or when only
    one episode is requested, where batching has no benefit).
    """
    _set_flat_params(policy, perturbed_params)

    if hasattr(policy, "forward_logits_batch") and n_eval_episodes > 1:
        return _evaluate_perturbation_batched(policy, env, n_eval_episodes, max_turns)
    return _evaluate_perturbation_serial(policy, env, n_eval_episodes, max_turns)


def es_gradient_estimate_wordle(
    policy,
    env,
    N: int = 20,
    sigma: float = 0.05,
    n_eval_episodes: int = 3,
    max_turns: int = 6,
    rank_fitness: bool = False,
    fitness_objective: str = "return",
    win_fitness_scale: float = 5.0,
    antithetic: bool = False,
    common_random_numbers: bool = False,
    baseline_subtract: bool = False,
    per_iter_secret_subset_size: Optional[int] = None,
) -> Tuple[torch.Tensor, float, List[float], float, float, List[float]]:
    """
    Estimate gradient using Evolution Strategies for Wordle.
    
    Algorithm:
        1. Sample N perturbations ε_i ~ N(0, I)
        2. Evaluate fitness R(θ + σε_i) for each perturbation
        3. Estimate gradient: ∇J ≈ (1/Nσ) Σ R(θ + σε_i) · ε_i
    
    Args:
        policy: WordleDiscretePolicy to optimize
        env: WordleEnvironmentWrapper
        N: Population size (number of perturbations)
        sigma: Noise scale
        n_eval_episodes: Episodes per perturbation evaluation
        max_turns: Max turns per episode (6 for Wordle)
        rank_fitness: If True, use centered ranks instead of z-scoring (often better for small N).
            Ignored when ``baseline_subtract=True``.
        fitness_objective: What ES maximizes per perturbation:
            ``"return"`` — mean episode return (default; partial credit inflates Fitness).
            ``"win"`` — mean win rate only (sparse; often needs rank_fitness and larger N).
            ``"win_plus_return"`` — ``win_fitness_scale * win_rate + mean_return`` so wins
            dominate but return still differentiates when everyone loses.
        win_fitness_scale: Multiplier on win rate for ``win_plus_return`` (default 5.0).
        antithetic: If True, sample ``N // 2`` noise vectors and evaluate both ``+ε`` and ``−ε``.
            Requires ``N`` to be even. Pure variance reduction on the ES gradient estimate.
        common_random_numbers: If True, snapshot the Python/numpy/torch RNG state once at
            the start of the estimate and restore it before each perturbation's evaluation,
            so every population member faces the same secret words and sampling draws. The
            outer RNG state is restored (and advanced once per call) before returning.
        baseline_subtract: If True, replace the rank/z-score fitness shaping with raw
            baseline-subtracted fitness (PGPE-lite). For non-antithetic populations the
            shaped fitness is ``F - mean(F)``; for antithetic populations it is the per-pair
            difference ``F_+i - F_-i`` (assigned with sign ``+`` to the ``+ε`` member and
            ``-`` to the ``-ε`` member). No std normalization. The motivation: when most
            population members tie at "lost everything" (the dominant regime when
            ``wins/N`` is small), rank-fitness collapses the bulk to identical ranks and
            then ``std(ranks)`` is dominated by the few discriminating positions, which
            *renormalizes away* the magnitude of the win signal that
            ``win_fitness_scale`` injects. Raw baseline subtraction preserves that
            magnitude — a member that wins gets fitness ``~ win_fitness_scale + return``
            while the bulk gets ``~ return``, so the centered signal is sharply peaked
            on the winners. Takes precedence over ``rank_fitness`` when both are True.
        per_iter_secret_subset_size: If set to ``k``, draw ``k`` secrets uniformly at
            random (without replacement) from the env's current target pool at the START
            of this iteration and temporarily restrict the iteration's secret pool to
            that subset. Under CRN, every population member then plays the same ``k``
            secrets, so each member sees each subset secret ~``n_eval_episodes / k``
            times in expectation -- this is the "mini-batch ES under CRN" regime that
            unlocks the signal-density bottleneck identified in week 12 (critiqueWeek12.md
            and docs/llm_exploration/week12_log.md). The env's target pool is restored
            before the function returns, regardless of success. ``None`` (default) keeps
            the env's full target pool for the iteration (legacy behavior). Requires the
            env to be ``_LocalWordleDataset``-backed (which is the case for every code
            path that goes through ``WordleEnvironmentWrapper.set_target_pool`` --
            Prime Intellect's dataset-of-prompts is not supported and will raise).
            When ``k`` is >= the current pool size, the call is a no-op on the pool.

    Returns:
        gradient: Estimated gradient (flattened parameter vector)
        avg_fitness: Average **optimization** fitness across population (same scale as objective)
        fitness_values: List of fitness values used for ES
        avg_es_win: Mean win rate (0–1) across the N perturbation rollouts
        pop_fitness_std: Std dev of per-member fitness (shows ES population spread)
        es_win_rates: Per-member win rates (list of length N), in population order. Used by
            ``train_es_wordle`` to compute the ``win_count`` diagnostic (number of population
            members that won at least one episode this iter); when 0 or 1, the rank vector
            is effectively noise.
    """
    # Get flattened trainable parameters (frozen layers excluded)
    params = torch.cat([p.flatten() for p in _trainable_params(policy)])
    policy_device = params.device
    n_params = params.shape[0]
    
    if fitness_objective not in ("return", "win", "win_plus_return"):
        raise ValueError(
            f"fitness_objective must be 'return', 'win', or 'win_plus_return', got {fitness_objective!r}"
        )

    if antithetic and (N % 2 != 0):
        raise ValueError(
            f"antithetic=True requires an even population size, got N={N}."
        )

    if antithetic:
        half = N // 2
        base_epsilons = [torch.randn(n_params, device=policy_device) for _ in range(half)]
        perturbations: List[torch.Tensor] = []
        for eps in base_epsilons:
            perturbations.append(eps)
            perturbations.append(-eps)
    else:
        perturbations = [torch.randn(n_params, device=policy_device) for _ in range(N)]

    # Mini-batch ES under CRN: temporarily subset the env's secret pool to ``k``
    # words for this iteration. Under CRN every member sees the same ``k``
    # secrets, so each secret is revisited ~n_eval_episodes/k times per member
    # per iter -- the signal-density regime Test B identified as the working
    # regime for ES on Wordle. Installed BEFORE the CRN snapshot so the snapshot
    # captures the post-subset numpy/torch state (otherwise restoring RNG before
    # each member would re-roll the subset across members, defeating CRN). Must
    # be restored in a ``finally`` so an exception in the eval loop doesn't
    # leave the env pinned to a subset across future iterations.
    original_targets: Optional[List[str]] = None
    if per_iter_secret_subset_size is not None:
        prime = getattr(env, "prime_env", None)
        ds = getattr(prime, "dataset", None) if prime is not None else None
        if ds is None or not hasattr(ds, "targets"):
            raise ValueError(
                "per_iter_secret_subset_size requires the env to be backed by a "
                "_LocalWordleDataset (WordleEnvironmentWrapper with a target_pool "
                "or fallen back to the bundled answer list). Prime Intellect "
                "dataset-of-prompts envs are not supported by this feature."
            )
        k = int(per_iter_secret_subset_size)
        if k < 1:
            raise ValueError(
                f"per_iter_secret_subset_size must be >= 1, got {k}."
            )
        original_targets = list(ds.targets)
        if k < len(original_targets):
            subset_idx = np.random.choice(
                len(original_targets), size=k, replace=False
            )
            subset_words = [original_targets[int(i)] for i in subset_idx]
            env.set_target_pool(subset_words)
        # else: k >= pool size -> no-op on the pool; original_targets is still
        # tracked so the ``finally`` re-install is safe.

    try:
        # Snapshot RNG *after* sampling perturbations (and *after* the optional
        # subset draw above) so each iteration's CRN stream depends on the
        # current outer RNG but is identical across population members.
        crn_snapshot: Optional[dict] = None
        outer_snapshot: Optional[dict] = None
        if common_random_numbers:
            outer_snapshot = _snapshot_rng_state(policy_device)
            crn_snapshot = outer_snapshot

        fitness_values: List[float] = []
        es_win_rates: List[float] = []

        for epsilon in perturbations:
            if crn_snapshot is not None:
                _restore_rng_state(crn_snapshot, policy_device)

            mean_ret, wr = _evaluate_perturbation(
                policy, env, params + sigma * epsilon, n_eval_episodes, max_turns
            )
            es_win_rates.append(wr)
            if fitness_objective == "return":
                fitness_values.append(mean_ret)
            elif fitness_objective == "win":
                fitness_values.append(float(wr))
            else:
                fitness_values.append(win_fitness_scale * float(wr) + mean_ret)
    finally:
        if original_targets is not None:
            # Restore the env's full target pool so downstream callers
            # (train_es_wordle's periodic eval, train_curriculum's stage
            # bookkeeping) see the pool they installed.
            env.set_target_pool(original_targets)

    if outer_snapshot is not None:
        # Advance every RNG that env.reset() / policy sampling actually consumes
        # inside an iteration, so the *next* iteration's CRN snapshot represents
        # a fresh stream:
        #   - WordleEnvironmentWrapper.reset() draws secrets via Python `random`
        #     (random.randint on the Prime path, random.choice on the mock path).
        #   - Policy sampling draws via torch (CUDA when policy is on cuda).
        # Without these explicit advances, restoring `outer_snapshot` would put
        # Python/numpy back to their pre-iteration state and the next snapshot
        # would capture the same Python state -> identical secret words every
        # ES iteration (CRN's whole point is intra-iteration sharing, not
        # inter-iteration freezing).
        _restore_rng_state(outer_snapshot, policy_device)
        _ = random.random()
        _ = np.random.rand()
        _ = torch.randn(1, device=policy_device)
        if torch.cuda.is_available() and torch.device(policy_device).type == "cuda":
            # Defensive: nothing in the current eval path reads torch CPU RNG,
            # but advance it too so future CPU-side samplers don't silently
            # re-introduce the inter-iteration freeze bug.
            _ = torch.randn(1, device="cpu")

    # Restore original parameters
    _set_flat_params(policy, params)
    
    # Compute gradient estimate
    fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32, device=policy_device)
    perturbations_tensor = torch.stack(perturbations)

    # Fitness shaping. Three mutually-exclusive branches, in priority order:
    #   1. baseline_subtract=True (overrides everything): raw centered fitness
    #      with NO std normalization. Antithetic: per-pair diff F_+i - F_-i.
    #      Non-antithetic: F - mean(F). This preserves the magnitude of the
    #      win signal (win_fitness_scale * win_rate) that the rank/z-score
    #      branches renormalize away when most members tie at "lost everything".
    #   2. rank_fitness=True: centered ranks, std-normalized. Robust to outlier
    #      fitness values but loses magnitude information.
    #   3. default: z-score (subtract mean, divide by std).
    #
    # When antithetic=True, perturbations are interleaved as
    #   [+ε_0, -ε_0, +ε_1, -ε_1, ...]
    # so the natural variance-reduced statistic is the per-pair difference
    # F_+i - F_-i (the pair mean cancels exactly). Ranking globally over all N
    # members re-introduces the pair-mean noise. For the rank-fitness branch we
    # rank the pair differences across N/2 pairs and assign +rank / -rank to the
    # two members of each pair; for the z-score branch we subtract the per-pair
    # mean before standardizing.
    if antithetic:
        half = N // 2
        f_plus = fitness_tensor[0::2]   # F_+i for i in [0, half)
        f_minus = fitness_tensor[1::2]  # F_-i for i in [0, half)
        if baseline_subtract:
            # PGPE-lite: per-pair diff is already a within-pair baseline-subtracted
            # signal. Skip both the rank transform and the std normalization so the
            # magnitude of the win signal survives into the gradient. The few pairs
            # with one winner and one loser dominate; the many pairs where both
            # members lost have diff ≈ 0 and contribute negligibly (instead of
            # being lifted to a fixed |rank| by the rank transform, where they
            # would inject noise weighted by the std denominator).
            diffs = f_plus - f_minus
            fitness_normalized = torch.empty(N, device=policy_device, dtype=torch.float32)
            fitness_normalized[0::2] = diffs
            fitness_normalized[1::2] = -diffs
        elif rank_fitness:
            diffs = f_plus - f_minus
            order = torch.argsort(diffs, descending=True)
            pair_ranks = torch.zeros(half, device=policy_device, dtype=torch.float32)
            for rank, idx in enumerate(order):
                pair_ranks[idx] = float(half - 1 - rank)
            pair_centered = pair_ranks - pair_ranks.mean()
            ps = pair_centered.std()
            if ps > 1e-8:
                pair_centered = pair_centered / ps
            fitness_normalized = torch.empty(N, device=policy_device, dtype=torch.float32)
            fitness_normalized[0::2] = pair_centered
            fitness_normalized[1::2] = -pair_centered
        else:
            pair_mean = 0.5 * (f_plus + f_minus)
            centered = torch.empty(N, device=policy_device, dtype=torch.float32)
            centered[0::2] = f_plus - pair_mean
            centered[1::2] = f_minus - pair_mean
            fitness_std = centered.std()
            if fitness_std > 1e-8:
                fitness_normalized = (centered - centered.mean()) / fitness_std
            else:
                fitness_normalized = centered - centered.mean()
    elif baseline_subtract:
        # Non-antithetic baseline subtraction: raw mean-centered fitness, no std
        # normalization. Same magnitude-preservation rationale as the antithetic
        # branch above.
        fitness_normalized = fitness_tensor - fitness_tensor.mean()
    elif rank_fitness:
        order = torch.argsort(fitness_tensor, descending=True)
        ranks = torch.zeros(N, device=policy_device, dtype=torch.float32)
        for rank, idx in enumerate(order):
            ranks[idx] = float(N - 1 - rank)
        fitness_normalized = ranks - ranks.mean()
        rs = fitness_normalized.std()
        if rs > 1e-8:
            fitness_normalized = fitness_normalized / rs
    else:
        fitness_std = fitness_tensor.std()
        if fitness_std > 1e-8:
            fitness_normalized = (fitness_tensor - fitness_tensor.mean()) / fitness_std
        else:
            fitness_normalized = fitness_tensor - fitness_tensor.mean()
    
    # Gradient estimate: (1/Nσ) Σ F_i · ε_i
    gradient = (perturbations_tensor.T @ fitness_normalized) / (N * sigma)
    
    avg_es_win = float(np.mean(es_win_rates))
    pop_fitness_std = float(np.std(fitness_values)) if len(fitness_values) > 1 else 0.0
    return gradient, fitness_tensor.mean().item(), fitness_values, avg_es_win, pop_fitness_std, es_win_rates


def train_es_wordle(
    policy,
    env,
    N: int = 20,
    sigma: float = 0.05,
    alpha: float = 0.01,
    n_iterations: int = 100,
    n_eval_episodes: int = 3,
    max_turns: int = 6,
    eval_every: int = 10,
    verbose: bool = True,
    normalize_gradient: bool = False,
    eval_n_episodes: int = 20,
    rank_fitness: bool = False,
    eval_deterministic: bool = True,
    fitness_objective: str = "win_plus_return",
    win_fitness_scale: float = 5.0,
    antithetic: bool = False,
    common_random_numbers: bool = False,
    ema_beta: float = 0.0,
    env_eval: Optional[Any] = None,
    probe_n_episodes: int = 32,
    probe_seed: int = 1234567,
    baseline_subtract: bool = False,
    per_iter_secret_subset_size: Optional[int] = None,
    restore_best_at_stage_end: bool = False,
    eval_stochastic_every: Optional[int] = None,
) -> Dict[str, List]:
    """
    Train policy using Evolution Strategies on Wordle.
    
    Args:
        policy: WordleDiscretePolicy to train
        env: WordleEnvironmentWrapper used for ES rollouts (training secret pool)
        N: Population size
        sigma: Noise scale
        alpha: Learning rate
        n_iterations: Number of training iterations
        n_eval_episodes: Episodes per fitness evaluation
        max_turns: Max turns per episode (6 for Wordle)
        eval_every: Evaluate policy every N iterations
        verbose: Print progress
        normalize_gradient: If True, apply θ ← θ + α·ĝ/‖ĝ‖ (stable for high-dim heads); else θ ← θ + α·ĝ
        eval_n_episodes: Number of episodes for periodic eval rollouts (logging only)
        rank_fitness: If True, ES uses rank-normalized fitness (recommended when N is small)
        eval_deterministic: If False, periodic eval matches ES rollouts (stochastic actions).
            Argmax eval can show 0% success while fitness uses sampling from the same logits.
        fitness_objective: Passed to ``es_gradient_estimate_wordle`` (default ``win_plus_return``).
        win_fitness_scale: Weight on win rate in ``win_plus_return`` mode.
        antithetic: If True, ES uses ``N // 2`` antithetic pairs (requires ``N`` even).
        common_random_numbers: If True, every population member in an iteration sees the same
            env + policy RNG stream (reduces per-member fitness variance at fixed N).
        ema_beta: If > 0, apply Adam-style EMA momentum to the ES gradient estimate. The applied
            gradient is bias-corrected ``g_ema / (1 - β^(t+1))`` so the persistent component of
            successive estimates accumulates while their noise averages out. ``ema_beta=0`` (default)
            reproduces the original no-momentum behavior.
        env_eval: Optional separate env for periodic eval. When provided, periodic eval rollouts
            sample secrets from ``env_eval`` (e.g. a held-out pool) while ES rollouts continue to
            use ``env`` (the training pool). When None, periodic eval uses ``env`` (legacy behavior).
        probe_n_episodes: Number of fixed-seed eval episodes used by the per-step probe (the
            ``train_probe_delta`` diagnostic). The probe runs greedy rollouts on ``env_eval`` (or
            ``env`` if no ``env_eval``) before *and* after applying each ES step on a deterministic
            slate of secrets, so the delta isolates the policy update's effect from secret-sampling
            noise. Only triggered on eval iters (``iteration % eval_every == 0``) so it adds at
            most ``2 * probe_n_episodes`` rollouts per ``eval_every`` ES iters.
        probe_seed: Fixed RNG seed installed before each probe rollout so pre- and post-update
            probes face identical secrets across iterations. The outer RNG state is snapshotted
            and restored around the probe so it does not perturb ES rollouts.
        baseline_subtract: If True, ES uses raw baseline-subtracted fitness instead of the
            rank/z-score branches. Recommended when ``wins/N`` is small per iteration (the
            "most members tie at lost-everything" regime), where rank-fitness collapses the
            tied bulk to a constant rank and ``std(ranks)`` renormalizes away the magnitude
            of the win signal that ``win_fitness_scale`` injects. Takes precedence over
            ``rank_fitness``. See ``es_gradient_estimate_wordle`` for details.
        per_iter_secret_subset_size: If set to ``k``, each ES iteration draws ``k`` secrets
            uniformly at random from the env's current target pool and restricts that
            iteration's fitness eval to those ``k`` secrets (mini-batch ES under CRN).
            Unlocks the signal-density regime when the global secret pool is larger than
            ``n_eval_episodes`` would otherwise allow. The env's full pool is restored
            before each iteration's periodic-eval rollouts (so eval_success still
            measures the full pool). ``None`` (default) keeps the legacy behavior.
            See ``es_gradient_estimate_wordle`` for the underlying implementation.
        restore_best_at_stage_end: If True, track the best-by-greedy-eval iterate during
            this ES run (``best_eval_success``, ``best_iter``, ``best_state`` as a CPU
            clone of ``policy.state_dict()``) and return it via the history dict under
            ``history["best_iter"]`` (1-element list), ``history["best_eval_success"]``
            (1-element list), and ``history["best_state"]`` (the actual state dict). The
            caller (e.g. ``train_curriculum``) is responsible for ``load_state_dict``-ing
            it back into the policy at the end of the stage. The CPU clone is only
            allocated when this flag is True -- the legacy default (False) has zero
            memory / runtime overhead. Rationale: under rotating mini-batch objectives
            (``per_iter_secret_subset_size`` active), ALPHA calibrated at iter 0 is
            tuned to a stationary objective and can overshoot when the objective rotates
            each iter; best-iter restore decouples "did ES find a good iterate?" from
            "did ES converge?".
        eval_stochastic_every: If set to ``m``, also run ``quick_eval_success(...,
            stochastic=True)`` on the same ``eval_env`` at every iter where the greedy
            eval fires AND ``iteration % m == 0`` (typically ``m == eval_every``). Results
            are appended to ``history["eval_success_stochastic"]``. The stochastic eval
            is wrapped in the same RNG snapshot/restore as ``_run_probe`` so it does not
            perturb the outer ES stream. Deterministically seeded (``probe_seed + 7000 +
            iteration``) so across-iter comparisons are apples-to-apples. ``None``
            (default) skips the stochastic eval entirely; ``history["eval_success_stochastic"]``
            is still initialized as an empty list so callers can read it unconditionally.

    Returns:
        history: Dictionary with training history
    """
    eval_env = env_eval if env_eval is not None else env
    # Flattened trainable parameters
    params = torch.cat([p.flatten() for p in _trainable_params(policy)])
    params_init = params.clone()

    # Eval checkpoints (every eval_every) — kept for backward compatibility
    history = {
        "iteration": [],
        "avg_fitness": [],
        "eval_reward": [],
        "eval_success": [],
        "eval_turns": [],
        "gradient_norm": [],
    }
    # Every ES step — use this to *see* optimization (param drift, fitness noise, etc.)
    history["train_iter"] = []
    history["train_fitness"] = []
    history["train_es_win"] = []
    history["train_grad_norm"] = []
    history["param_drift"] = []
    history["pop_fitness_std"] = []
    # Cosine similarity of successive *raw* ES gradients — the real "is there signal?" plot.
    # NaN for iteration 0 (no previous gradient to compare against).
    history["train_grad_cos"] = []
    # ES signal diagnostics. `train_ess_rank` is the number of unique fitness values across the
    # population — when << N, ties dominate the rank ordering and the ES gradient is
    # variance-dominated noise. `train_win_count` is the number of population members that
    # won at least one episode; 0 or 1 means the rank vector is essentially noise.
    # `train_probe_delta` is the change in greedy success on a fixed probe slate caused by this
    # iter's ES step (NaN on non-eval iters); it is the most direct "did this step help?" signal.
    history["train_ess_rank"] = []
    history["train_win_count"] = []
    history["train_probe_delta"] = []
    # Populated on every iter where greedy eval fires and
    # ``iteration % eval_stochastic_every == 0``; empty list when
    # ``eval_stochastic_every is None``. Same cadence as history["eval_success"]
    # when eval_stochastic_every == eval_every, so downstream plots can zip them.
    history["eval_success_stochastic"] = []
    # Populated once per call when ``restore_best_at_stage_end`` is True (single-element
    # lists); left empty otherwise. ``best_state`` is the actual CPU-cloned state_dict
    # the caller needs to ``load_state_dict`` back -- callers that persist ``history``
    # should pop this key first (it's a full model copy).
    history["best_iter"] = []
    history["best_eval_success"] = []

    use_ema = ema_beta > 0.0
    g_ema: Optional[torch.Tensor] = None
    prev_raw_gradient: Optional[torch.Tensor] = None
    policy_device = next(policy.parameters()).device

    # Best-iter tracking (only allocate the state clone when the flag is True so
    # the legacy path has no memory overhead). best_iter = -1 sentinel for
    # "never saw an eval checkpoint" (should not happen with eval_every < n_iters).
    best_eval_success: float = float("-inf")
    best_iter_idx: int = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def _run_probe() -> float:
        """Greedy rollout on a fixed slate of secrets. Snapshots+restores RNG so the probe
        is reproducible across iterations and does not perturb the surrounding ES stream.
        """
        if probe_n_episodes <= 0:
            return float("nan")
        rng_snap = _snapshot_rng_state(policy_device)
        try:
            random.seed(probe_seed)
            np.random.seed(probe_seed)
            torch.manual_seed(probe_seed)
            if torch.cuda.is_available() and policy_device.type == "cuda":
                torch.cuda.manual_seed_all(probe_seed)
            if hasattr(policy, "forward_logits_batch") and probe_n_episodes > 1:
                _r, successes, _t = _rollout_batched(
                    policy, eval_env, probe_n_episodes, max_turns, deterministic=True
                )
            else:
                successes = []
                policy.eval()
                with torch.no_grad():
                    for _ in range(probe_n_episodes):
                        s = eval_env.reset()
                        d = False
                        info: dict = {}
                        t = 0
                        while not d and t < max_turns:
                            emb = eval_env.get_state_embedding(s)
                            xml, _lp = policy.format_action_xml(s, emb, deterministic=True)
                            s, _r, d, info = eval_env.step(xml)
                            t += 1
                        successes.append(float(info.get("correct_answer", 0.0)))
            return float(np.mean(successes)) if successes else float("nan")
        finally:
            _restore_rng_state(rng_snap, policy_device)

    for iteration in range(n_iterations):
        # ES gradient step
        (
            gradient,
            avg_fitness,
            fitness_values,
            avg_es_win,
            pop_fitness_std,
            es_win_rates,
        ) = es_gradient_estimate_wordle(
            policy,
            env,
            N=N,
            sigma=sigma,
            n_eval_episodes=n_eval_episodes,
            max_turns=max_turns,
            rank_fitness=rank_fitness,
            fitness_objective=fitness_objective,
            win_fitness_scale=win_fitness_scale,
            antithetic=antithetic,
            common_random_numbers=common_random_numbers,
            baseline_subtract=baseline_subtract,
            per_iter_secret_subset_size=per_iter_secret_subset_size,
        )

        # Cosine between raw ĝ_t and ĝ_{t-1}: positive values indicate a consistent
        # direction across iterations (real signal); near-zero means diffusion.
        if prev_raw_gradient is not None:
            g_norm = gradient.norm()
            prev_norm = prev_raw_gradient.norm()
            if g_norm > 1e-8 and prev_norm > 1e-8:
                grad_cos = float(torch.dot(gradient, prev_raw_gradient) / (g_norm * prev_norm))
            else:
                grad_cos = float("nan")
        else:
            grad_cos = float("nan")
        prev_raw_gradient = gradient.detach().clone()

        # ESS for the rank ordering: number of unique fitness values across the population.
        # Rounded to 6 decimals to merge values that differ only in float noise. With
        # rank_fitness=True and small n_eval_episodes, ties dominate when most members lose
        # every episode (all wr=0) and `ess_rank` collapses toward 1, which directly explains
        # `cos(ĝ) ≈ 0` -- the ES gradient is then essentially a sum of noise vectors with
        # near-uniform "fitness" weights.
        ess_rank = len({round(float(f), 6) for f in fitness_values})
        # Number of population members that won at least one episode this iter.
        win_count = sum(1 for wr in es_win_rates if wr > 0.0)

        # Apply EMA momentum (with Adam-style bias correction) so persistent signal
        # accumulates across iterations while noise averages out.
        if use_ema:
            if g_ema is None:
                g_ema = torch.zeros_like(gradient)
            g_ema = ema_beta * g_ema + (1.0 - ema_beta) * gradient
            bias_correction = 1.0 - (ema_beta ** (iteration + 1))
            applied_gradient = g_ema / bias_correction
        else:
            applied_gradient = gradient

        # Update parameters (optional unit-norm step: avoids huge ‖ĝ‖ when dim(θ) is large)
        grad_norm = applied_gradient.norm().item()
        update = alpha * applied_gradient
        if normalize_gradient and grad_norm > 1e-8:
            update = alpha * applied_gradient / applied_gradient.norm()

        # Pre/post probe on a fixed-seed eval slate, only on eval iters (cost is
        # 2 * probe_n_episodes greedy rollouts per `eval_every` ES iters). Pre-probe
        # uses the *current* params; post-probe uses params after the update. Same
        # `probe_seed` => identical secret slate => the delta isolates the policy
        # update from secret-sampling noise.
        is_eval_iter = (iteration % eval_every == 0) or (iteration == n_iterations - 1)
        if is_eval_iter and probe_n_episodes > 0:
            pre_probe = _run_probe()
        else:
            pre_probe = float("nan")

        params = params + update
        _set_flat_params(policy, params)
        step_norm = update.norm().item()
        param_drift = (params - params_init).norm().item()

        if is_eval_iter and probe_n_episodes > 0:
            post_probe = _run_probe()
            probe_delta = (
                post_probe - pre_probe
                if (post_probe == post_probe and pre_probe == pre_probe)
                else float("nan")
            )
        else:
            probe_delta = float("nan")

        history["train_iter"].append(iteration)
        history["train_fitness"].append(avg_fitness)
        history["train_es_win"].append(avg_es_win)
        history["train_grad_norm"].append(grad_norm)
        history["param_drift"].append(param_drift)
        history["pop_fitness_std"].append(pop_fitness_std)
        history["train_grad_cos"].append(grad_cos)
        history["train_ess_rank"].append(ess_rank)
        history["train_win_count"].append(win_count)
        history["train_probe_delta"].append(probe_delta)

        # Periodic evaluation (full rollout stats; slow when eval_n_episodes is large).
        # When the policy exposes ``forward_logits_batch`` we use the same
        # lockstep-batched rollout the ES fitness loop uses; per-episode
        # semantics (per-episode reward / success / turns appended in order,
        # greedy/sample switch via ``deterministic``) are preserved. With
        # eval_every=1 + eval_n_episodes=50 + Gemma-3-1b this is otherwise the
        # dominant wall-clock in the training loop.
        if iteration % eval_every == 0 or iteration == n_iterations - 1:
            if hasattr(policy, "forward_logits_batch") and eval_n_episodes > 1:
                eval_rewards, eval_successes, eval_turn_counts = _rollout_batched(
                    policy,
                    eval_env,
                    eval_n_episodes,
                    max_turns,
                    deterministic=eval_deterministic,
                )
            else:
                eval_rewards = []
                eval_successes = []
                eval_turn_counts = []

                policy.eval()
                with torch.no_grad():
                    for _ in range(eval_n_episodes):
                        state = eval_env.reset()
                        episode_reward = 0
                        done = False
                        turns = 0
                        info: dict = {}

                        while not done and turns < max_turns:
                            state_embedding = eval_env.get_state_embedding(state)

                            if hasattr(policy, 'format_action_xml'):
                                action_xml, _ = policy.format_action_xml(
                                    state, state_embedding, deterministic=eval_deterministic
                                )
                            else:
                                action_idx, _ = policy.get_action(
                                    state_embedding, deterministic=eval_deterministic
                                )
                                word = policy.vocab.action_to_word(action_idx)
                                action_xml = f"<guess>{word}</guess>"

                            state, reward, done, info = eval_env.step(action_xml)
                            episode_reward += reward
                            turns += 1

                        eval_rewards.append(episode_reward)
                        eval_successes.append(float(info.get('correct_answer', 0.0)))
                        eval_turn_counts.append(turns)

            eval_reward = float(np.mean(eval_rewards))
            eval_success = float(np.mean(eval_successes))
            eval_turns = float(np.mean(eval_turn_counts))
            
            history['iteration'].append(iteration)
            history['avg_fitness'].append(avg_fitness)
            history['eval_reward'].append(eval_reward)
            history['eval_success'].append(eval_success)
            history['eval_turns'].append(eval_turns)
            history['gradient_norm'].append(grad_norm)

            # Stochastic-sampling eval companion to greedy eval_success. Fires at the
            # same cadence as the greedy eval when ``eval_stochastic_every`` is set
            # (typically == eval_every so the two lists zip for post-hoc plotting).
            # Wrapped in RNG snapshot/restore so the stochastic eval does not perturb
            # the outer ES stream; deterministically seeded so across-iter comparisons
            # face the same secret slate.
            if (
                eval_stochastic_every is not None
                and (iteration % eval_stochastic_every == 0 or iteration == n_iterations - 1)
            ):
                try:
                    from .wordle_gpt2_warmstart import quick_eval_success as _qes
                except ImportError:
                    try:
                        from wordle_gpt2_warmstart import quick_eval_success as _qes
                    except ImportError:
                        _qes = None  # type: ignore
                if _qes is not None and eval_n_episodes > 0:
                    _stoch_snap = _snapshot_rng_state(policy_device)
                    try:
                        _stoch_seed = int(probe_seed) + 7000 + int(iteration)
                        random.seed(_stoch_seed)
                        np.random.seed(_stoch_seed)
                        torch.manual_seed(_stoch_seed)
                        if torch.cuda.is_available() and policy_device.type == "cuda":
                            torch.cuda.manual_seed_all(_stoch_seed)
                        _stoch_success = float(
                            _qes(
                                policy,
                                eval_env,
                                n_episodes=eval_n_episodes,
                                stochastic=True,
                                max_turns=max_turns,
                            )
                        )
                    finally:
                        _restore_rng_state(_stoch_snap, policy_device)
                    history['eval_success_stochastic'].append(_stoch_success)

            # Best-iter tracking. Only allocate the CPU state clone when the flag is
            # True (legacy path has zero memory overhead). Ties broken by "earliest
            # wins" (strict > below) so a later regression to the same value doesn't
            # churn the clone. eval_success is greedy; callers grading against
            # ``best_greedy`` are comparing apples-to-apples against post_ws_greedy.
            if restore_best_at_stage_end and eval_success > best_eval_success:
                best_eval_success = eval_success
                best_iter_idx = iteration
                best_state = {
                    k: v.detach().to("cpu").clone()
                    for k, v in policy.state_dict().items()
                }

            _cos_str = "  n/a" if grad_cos != grad_cos else f"{grad_cos:+.2f}"
            _dprobe_str = "  n/a" if probe_delta != probe_delta else f"{probe_delta:+.1%}"
            # Companion stochastic eval printout (only when the stochastic eval fired
            # on this iteration). Empty string keeps the legacy line format intact
            # when eval_stochastic_every is None.
            if (
                eval_stochastic_every is not None
                and history['eval_success_stochastic']
                and (iteration % eval_stochastic_every == 0 or iteration == n_iterations - 1)
            ):
                _stoch_str = f" | Stoch: {history['eval_success_stochastic'][-1]:5.1%}"
            else:
                _stoch_str = ""
            if verbose:
                _ev = "greedy" if eval_deterministic else "stoch"
                _fl = {"return": "ret", "win": "win", "win_plus_return": "win+ret"}.get(
                    fitness_objective, fitness_objective
                )
                print(
                    f"Iter {iteration:4d} | "
                    f"Fit({_fl}): {avg_fitness:6.3f} | "
                    f"ES_win: {avg_es_win:5.1%} | "
                    f"popσ: {pop_fitness_std:.4f} | "
                    f"Eval Reward: {eval_reward:6.3f} | "
                    f"Success: {eval_success:5.1%} ({_ev}){_stoch_str} | "
                    f"Turns: {eval_turns:4.1f} | "
                    f"Grad‖: {grad_norm:.2f} | "
                    f"Step‖: {step_norm:.4f} | "
                    f"cos(ĝ): {_cos_str} | "
                    f"‖θ-θ₀‖: {param_drift:.2f} | "
                    f"ess: {ess_rank}/{N} | "
                    f"wins: {win_count}/{N} | "
                    f"dprobe: {_dprobe_str}"
                )
        elif verbose:
            _cos_str = "  n/a" if grad_cos != grad_cos else f"{grad_cos:+.2f}"
            _fl = {"return": "ret", "win": "win", "win_plus_return": "win+ret"}.get(
                fitness_objective, fitness_objective
            )
            print(
                f"Iter {iteration:4d} | Fit({_fl}): {avg_fitness:6.3f} | "
                f"ES_win: {avg_es_win:5.1%} | "
                f"popσ: {pop_fitness_std:.4f} | "
                f"Grad‖: {grad_norm:.2f} | Step‖: {step_norm:.4f} | "
                f"cos(ĝ): {_cos_str} | "
                f"‖θ-θ₀‖: {param_drift:.2f} | "
                f"ess: {ess_rank}/{N} | wins: {win_count}/{N} | (no eval)"
            )

    # Emit best-iter bookkeeping only when the tracker was active. Keeps the
    # history dict's shape identical to the legacy path when the flag is off
    # (the two keys were initialized to [] and remain []).
    if restore_best_at_stage_end and best_iter_idx >= 0:
        history['best_iter'].append(int(best_iter_idx))
        history['best_eval_success'].append(float(best_eval_success))
        # best_state is the actual CPU state dict -- stored under a separate key
        # (not a list) so callers don't accidentally pickle it as part of a
        # standard per-eval metric. ``train_curriculum`` pops this after
        # load_state_dict so it doesn't leak into ``combined``.
        history['best_state'] = best_state

    return history


# Keys produced by ``train_es_wordle`` that are appended once per ES iteration.
_PER_ITER_KEYS = (
    "train_iter",
    "train_fitness",
    "train_es_win",
    "train_grad_norm",
    "param_drift",
    "pop_fitness_std",
    "train_grad_cos",
    "train_ess_rank",
    "train_win_count",
    "train_probe_delta",
)
# Keys produced once per eval checkpoint.
_PER_EVAL_KEYS = (
    "iteration",
    "avg_fitness",
    "eval_reward",
    "eval_success",
    "eval_turns",
    "gradient_norm",
    # Optional: only populated when ``eval_stochastic_every`` is set. Empty list
    # under the legacy path; extend into ``combined`` is a no-op then. When the
    # stochastic cadence matches ``eval_every`` (the typical case) this list
    # zips with ``eval_success`` for greedy-vs-stochastic plots.
    "eval_success_stochastic",
)


def train_curriculum(
    policy,
    env,
    vocab_schedule: Sequence[int],
    *,
    n_iterations_per_stage: Any,
    warm_start_fn: Optional[Callable[..., Any]] = None,
    warm_start_steps: Union[int, Sequence[int]] = 0,
    warm_start_kwargs: Optional[Dict[str, Any]] = None,
    warm_start_seed_stride: int = 1000,
    verbose: bool = True,
    post_warm_start_eval_episodes: int = 50,
    post_warm_start_eval_deterministic: bool = True,
    env_eval: Optional[Any] = None,
    secret_holdout_frac: float = 0.0,
    expand_action_space: bool = True,
    holdout_mode: str = "episode",
    masked_eval_sanity_probe: bool = False,
    warm_start_max_post_ws_success: Optional[float] = None,
    restore_best_at_stage_end: bool = False,
    eval_stochastic_every: Optional[int] = None,
    **es_kwargs: Any,
) -> Dict[str, List]:
    """Curriculum ES: grow the policy/env vocabulary across stages.

    For each ``N_i`` in ``vocab_schedule``:

    1. If ``expand_action_space`` is True (default, legacy behavior),
       ``policy.expand_vocab(N_i)`` (no-op if the policy is already that size).
       If False, the policy's action space is left untouched (caller is
       responsible for sizing it before this call); only the env's secret
       pool is re-targeted per stage.
    2. ``env.set_target_pool(...)``: behavior depends on ``holdout_mode``.
       - ``"word"`` (legacy, opt-in): when ``secret_holdout_frac > 0`` the
         slice ``policy.words[:N_i]`` is partitioned into
         ``ws_pool = words[:-n_eval]`` and ``eval_pool = words[-n_eval:]``,
         with ``n_eval = max(2, round(N_i * secret_holdout_frac))``;
         ``env`` is set to ``ws_pool`` and ``env_eval`` (if provided) is set
         to ``eval_pool``. NB: under greedy ``argmax`` eval over the full
         action space, the eval-pool indices are never CE targets nor ES
         winners, so this metric is mathematically forced toward 0 even
         when the policy has learned Wordle. Use only for explicit
         out-of-vocab generalization experiments.
       - ``"episode"`` (default): ``ws_pool = eval_pool = stage_pool``;
         both ``env`` and ``env_eval`` (when provided) are set to the same
         pool. The held-out signal then comes from eval rollouts naturally
         drawing different secrets / first-guess prefixes from training
         rollouts (different RNG positions), so eval measures generalization
         to fresh game *episodes* rather than out-of-vocab words. The head
         can in principle argmax to any eval secret, so the metric is
         answerable by the architecture.
    3. If ``warm_start_fn`` is provided and the stage's training pool changed,
       run ``warm_start_fn(policy, env, n_steps=warm_start_steps, **warm_start_kwargs)``
       to seed the head on the training secret distribution.
    4. Quick post-warm-start eval using ``quick_eval_success``: held-out (on
       ``env_eval``) is the headline number, in-distribution (on ``env``) is
       logged for comparison so the memorization gap is visible per stage.
    5. ``train_es_wordle(policy, env, env_eval=env_eval, n_iterations=iters_for_stage, **es_kwargs)``.

    ``n_iterations_per_stage`` may be either:
        - an ``int`` (same iter budget for every stage), or
        - a sequence of ``int`` whose length matches ``vocab_schedule`` (per-stage
          iter budgets — useful for biasing iters toward the harder late stages).

    ``warm_start_steps`` may be either:
        - an ``int`` (same warm-start budget for every stage), or
        - a sequence of ``int`` whose length matches ``vocab_schedule`` (per-stage
          budgets — fixes the "50 Adam steps cannot fit a 1024-way head" failure
          mode by giving deeper stages proportionally more supervised steps).

    ``warm_start_seed_stride`` (default 1000) is added to the user-supplied
    ``warm_start_kwargs["seed"]`` per stage so the random pre-play sequence and
    the warm-start RNG state are *different* across stages. Without this, every
    stage's warm-start re-seeds the global Python/numpy/torch RNGs to the same
    value, which makes the start-of-stage ES iter strongly correlated across
    stages and obscures stage-to-stage progress. Set to 0 to disable.

    ``env_eval`` is an optional separate env used for periodic eval (both the
    post-warm-start probe and the eval rollouts inside ``train_es_wordle``).
    With ``secret_holdout_frac > 0`` it should be a fresh env identical in
    construction to ``env`` so per-stage ``set_target_pool`` calls can swap in
    the held-out pool without touching the training env.

    ``secret_holdout_frac`` controls the per-stage train/held-out split on the
    secret pool when ``holdout_mode="word"``. 0.0 reproduces the legacy
    no-holdout behavior. Values in ``(0, 0.5)`` partition the stage-k pool
    ``policy.words[:N_k]`` into a deterministic prefix (training secrets) and
    suffix (held-out secrets). Ignored when ``holdout_mode="episode"`` (in
    which case the train and eval pools are identical and held-out semantics
    come from RNG positions rather than from a vocab split).

    ``holdout_mode`` selects how the per-stage train / held-out split is
    constructed (see step 2 above). Defaults to ``"episode"``; pass
    ``"word"`` to restore the prior split-by-suffix behavior.

    ``warm_start_max_post_ws_success`` (default None): when set to a float in
    ``(0, 1]``, monitors post-warm-start in-distribution success per stage and
    suppresses warm-start in *all subsequent* stages once any stage exceeds the
    ceiling. The motivation: if the supervised warm-start already saturates the
    head (post-WS = 100%), ES has no headroom to improve and the cos(ĝ) signal
    is structurally zero. Capping at ~0.85 leaves ES ~15 percentage points of
    headroom per stage so the "is ES doing anything?" signal is observable.
    Defaults to None (no ceiling, legacy behavior).

    ``masked_eval_sanity_probe`` (default False): when True, after the
    post-warm-start eval each stage also runs a one-shot greedy eval on
    ``env_eval`` with logits *masked* to the indices in ``eval_pool`` (i.e.
    argmax restricted to the held-out / in-vocab indices). Requires
    ``quick_eval_success_masked`` from ``wordle_gpt2_warmstart``. Useful as a
    diagnostic in two ways:
       - In ``holdout_mode="word"``: confirms whether the unmasked 0% is a
         metric construction artifact (masked >> 0 means the policy can win
         on held-out secrets when argmax is restricted, so the prior
         unmasked 0% was forced by the disjoint-word eval, not a learning
         failure).
       - In ``holdout_mode="episode"``: shows what greedy success would be
         if the head were prevented from emitting words outside the stage
         vocabulary; gap vs. unmasked eval quantifies how much argmax mass
         leaks outside the in-vocab pool.

    ``expand_action_space`` (default True) controls whether each stage calls
    ``policy.expand_vocab(N_i)``. Set False when the policy is pre-built with
    its final action space (e.g. ``max_vocab_size=MAX_VOCAB``) and only the
    secret pool should grow under the curriculum.

    ``restore_best_at_stage_end`` (default False) — when True, each stage tracks
    the best-by-greedy-eval iterate (CPU-cloned ``policy.state_dict()``) during
    its ES run and ``load_state_dict``-s it back at the end of the stage.
    Appends the best iter (globally offset) to ``combined["best_iter"]`` and
    the corresponding greedy ``eval_success`` to ``combined["best_eval_success"]``.
    Motivation: under rotating mini-batch objectives (``per_iter_secret_subset_size``
    active) ALPHA calibrated at iter 0 assumes a stationary objective and can
    overshoot when the level sets rotate each iter; best-iter restore decouples
    "did ES find a good iterate?" from "did ES converge?". Default False
    preserves the legacy behavior and has zero memory overhead.

    ``eval_stochastic_every`` (default None) — when set to an int ``m``, each stage
    runs ``quick_eval_success(..., stochastic=True)`` on the stage's eval env
    on every iter where greedy eval fires AND ``iteration % m == 0`` (the
    typical call site sets ``m == eval_every`` so the stochastic list zips
    with the greedy one for post-hoc plots). Wrapped in RNG snapshot/restore
    so the stochastic eval does not perturb the outer ES stream, and
    deterministically seeded so across-iter comparisons are apples-to-apples.
    Results are accumulated into ``combined["eval_success_stochastic"]``.
    Default None skips the stochastic eval entirely (empty list; zero
    overhead).

    The combined history concatenates per-stage histories with iteration indices
    offset to be globally monotonic, plus extra keys:
        - ``stage_starts``: global iteration indices where each stage began.
        - ``stage_vocab_sizes``: the policy action_dim at each stage (constant
          when ``expand_action_space=False``).
        - ``stage_secret_pool_sizes``: the size of the stage's training
          secret pool (``ws_pool``).
        - ``stage_eval_pool_sizes``: the size of the stage's held-out secret
          pool (0 when ``secret_holdout_frac == 0`` or ``env_eval is None``).
        - ``post_warmstart_success``: greedy eval_success right after warm-start
          for each stage. Held-out value when both ``env_eval`` and a non-zero
          ``secret_holdout_frac`` are configured; otherwise in-distribution
          (legacy meaning).
        - ``post_warmstart_success_heldout``: held-out post-WS eval (NaN when
          ``env_eval is None`` or ``secret_holdout_frac == 0``).
        - ``post_warmstart_success_indist``: in-distribution post-WS eval on
          ``env`` (the training pool).
        - ``stage_post_warmstart_iter``: global iter index where each stage's
          post-warm-start eval was taken (= ``stage_starts[stage]``).
        - ``stage_holdout_mode``: the ``holdout_mode`` used for each stage
          (constant across stages but recorded per-stage so post-hoc plots
          can label the metric).
        - ``stage_masked_post_warmstart_success``: per-stage greedy eval on
          ``env_eval`` with logits masked to the in-vocab eval-pool indices,
          when ``masked_eval_sanity_probe=True``; NaN otherwise.

    LoRA adapters and the LM body persist across stages — only the linear head
    grows when ``expand_action_space=True``, with old rows preserved by
    ``WordleGPT2Policy.expand_vocab``.
    """
    if not 0.0 <= secret_holdout_frac < 0.5:
        raise ValueError(
            f"secret_holdout_frac must be in [0, 0.5), got {secret_holdout_frac}."
        )
    if holdout_mode not in ("episode", "word"):
        raise ValueError(
            f"holdout_mode must be 'episode' or 'word', got {holdout_mode!r}."
        )
    if holdout_mode == "word" and secret_holdout_frac > 0 and env_eval is None:
        raise ValueError(
            "holdout_mode='word' with secret_holdout_frac > 0 requires env_eval "
            "so the held-out secret pool can be installed without polluting training."
        )
    if not vocab_schedule:
        raise ValueError("vocab_schedule must contain at least one stage size.")

    if warm_start_max_post_ws_success is not None and not (
        0.0 < warm_start_max_post_ws_success <= 1.0
    ):
        raise ValueError(
            "warm_start_max_post_ws_success must be in (0, 1] when set, got "
            f"{warm_start_max_post_ws_success}."
        )

    if isinstance(n_iterations_per_stage, int):
        if n_iterations_per_stage <= 0:
            raise ValueError("n_iterations_per_stage must be positive.")
        per_stage_iters: List[int] = [int(n_iterations_per_stage)] * len(vocab_schedule)
    else:
        per_stage_iters = [int(x) for x in n_iterations_per_stage]
        if len(per_stage_iters) != len(vocab_schedule):
            raise ValueError(
                "n_iterations_per_stage as a sequence must have the same length as "
                f"vocab_schedule ({len(vocab_schedule)}); got {len(per_stage_iters)}."
            )
        if any(n <= 0 for n in per_stage_iters):
            raise ValueError(
                f"All per-stage iter counts must be positive, got {per_stage_iters}."
            )

    warm_start_kwargs = dict(warm_start_kwargs or {})

    if isinstance(warm_start_steps, int):
        if warm_start_steps < 0:
            raise ValueError("warm_start_steps must be non-negative.")
        per_stage_warm_steps: List[int] = [int(warm_start_steps)] * len(vocab_schedule)
    else:
        per_stage_warm_steps = [int(x) for x in warm_start_steps]
        if len(per_stage_warm_steps) != len(vocab_schedule):
            raise ValueError(
                "warm_start_steps as a sequence must have the same length as "
                f"vocab_schedule ({len(vocab_schedule)}); got {len(per_stage_warm_steps)}."
            )
        if any(n < 0 for n in per_stage_warm_steps):
            raise ValueError(
                f"All per-stage warm_start_steps must be non-negative, got {per_stage_warm_steps}."
            )

    base_warm_seed = int(warm_start_kwargs.get("seed", 0))

    # Optional imports: only used for the post-warm-start diagnostic.
    try:
        from .wordle_gpt2_warmstart import quick_eval_success  # type: ignore
    except ImportError:
        try:
            from wordle_gpt2_warmstart import quick_eval_success  # type: ignore
        except ImportError:
            quick_eval_success = None  # type: ignore

    quick_eval_success_masked = None  # type: ignore
    if masked_eval_sanity_probe:
        try:
            from .wordle_gpt2_warmstart import quick_eval_success_masked  # type: ignore
        except ImportError:
            try:
                from wordle_gpt2_warmstart import quick_eval_success_masked  # type: ignore
            except ImportError:
                quick_eval_success_masked = None  # type: ignore

    combined: Dict[str, List] = {k: [] for k in _PER_ITER_KEYS}
    combined.update({k: [] for k in _PER_EVAL_KEYS})
    combined["stage_starts"] = []
    combined["stage_vocab_sizes"] = []
    combined["stage_secret_pool_sizes"] = []
    combined["stage_eval_pool_sizes"] = []
    combined["post_warmstart_success"] = []
    combined["post_warmstart_success_heldout"] = []
    combined["post_warmstart_success_indist"] = []
    combined["stage_post_warmstart_iter"] = []
    combined["stage_holdout_mode"] = []
    combined["stage_masked_post_warmstart_success"] = []
    # Per-stage best-iterate bookkeeping. Populated only when
    # ``restore_best_at_stage_end`` is True; empty lists under the legacy path.
    # ``best_iter`` is stored as a globally-monotonic iteration index (offset
    # by ``iter_offset`` like other per-eval ``iteration`` entries) so it can
    # be compared against ``combined["iteration"]`` directly.
    combined["best_iter"] = []
    combined["best_eval_success"] = []

    iter_offset = 0
    prev_ws_pool: Optional[List[str]] = None
    # Tracks whether a previous stage's post-WS success exceeded
    # ``warm_start_max_post_ws_success``. Once tripped, subsequent stages skip
    # warm-start entirely so ES has headroom to demonstrate gain. The trip is
    # one-way (we don't re-enable warm-start on later, harder stages) because
    # carrying the warm-started head forward is the explicit design intent of
    # the curriculum -- only the *additional* CE budget per stage is suppressed.
    warm_start_ceiling_tripped = False
    for stage_idx, target_n in enumerate(vocab_schedule):
        prev_n = len(policy.words)
        if expand_action_space:
            new_n = policy.expand_vocab(int(target_n))
        else:
            new_n = len(policy.words)
        added = new_n - prev_n

        # Slice the stage's secret pool from the policy's word list. With a
        # constant action space (expand_action_space=False), policy.words is
        # already the full action set, and target_n picks the prefix used as
        # this stage's secret pool. With expand_action_space=True (legacy),
        # policy.words has just been grown to target_n above.
        stage_pool_size = min(int(target_n), len(policy.words))
        stage_pool = list(policy.words[:stage_pool_size])

        if holdout_mode == "word" and secret_holdout_frac > 0 and env_eval is not None:
            # Word-level holdout (legacy / opt-in): partition the stage pool into a
            # disjoint training prefix and held-out suffix. NB: under greedy argmax over
            # the full action space this measures out-of-vocab classification, not
            # Wordle generalization -- the head is never trained to emit eval-pool words.
            n_eval_pool = max(2, int(round(stage_pool_size * secret_holdout_frac)))
            n_eval_pool = min(n_eval_pool, max(stage_pool_size - 1, 1))
            ws_pool = stage_pool[:-n_eval_pool] if n_eval_pool > 0 else stage_pool
            eval_pool = stage_pool[-n_eval_pool:] if n_eval_pool > 0 else []
            env.set_target_pool(ws_pool)
            if eval_pool:
                env_eval.set_target_pool(eval_pool)
        elif holdout_mode == "episode":
            # Episode-level holdout: same vocab in train and eval. Train and eval rollouts
            # naturally consume different positions of the global RNG (different
            # `.reset()` calls), so eval secrets / first-guess prefixes differ from
            # training. The architecture *can* in principle argmax to any eval secret.
            ws_pool = stage_pool
            eval_pool = stage_pool  # for the masked-eval probe and eval_pool_size logging
            env.set_target_pool(ws_pool)
            if env_eval is not None:
                env_eval.set_target_pool(ws_pool)
        else:
            # holdout_mode == "word" but no holdout configured (frac == 0 or env_eval None):
            # fall back to the legacy "no holdout" behavior — train and eval share the pool.
            ws_pool = stage_pool
            eval_pool = []
            env.set_target_pool(ws_pool)
            if env_eval is not None:
                env_eval.set_target_pool(ws_pool)

        ws_steps_stage = per_stage_warm_steps[stage_idx]
        if warm_start_ceiling_tripped:
            # A prior stage already saturated post-WS success above the
            # configured ceiling; suppress further CE here so ES has room
            # to act. We zero the budget rather than skipping the step
            # entirely so the bookkeeping (ws_pool tracking, post-WS eval)
            # still runs and the per-stage history rows stay consistent.
            ws_steps_stage = 0
        ws_pool_changed = prev_ws_pool != ws_pool
        if verbose:
            if holdout_mode == "episode":
                pool_desc = (
                    f"secret_pool: {stage_pool_size} "
                    f"(episode-holdout; ws == eval == {len(ws_pool)})"
                )
            elif eval_pool:
                pool_desc = (
                    f"secret_pool: {stage_pool_size} "
                    f"(word-holdout; ws={len(ws_pool)}, eval={len(eval_pool)})"
                )
            else:
                pool_desc = f"secret_pool: {stage_pool_size}"
            if expand_action_space:
                action_desc = f"action_dim: {prev_n} -> {new_n} (+{added})"
            else:
                action_desc = f"action_dim: {new_n} (fixed)"
            print(
                f"\n=== Curriculum stage {stage_idx + 1}/{len(vocab_schedule)} "
                f"| {action_desc} | {pool_desc} "
                f"| iters: {per_stage_iters[stage_idx]} "
                f"| warm-start eps: {ws_steps_stage} ===",
                flush=True,
            )

        ran_warm_start = False
        # Run warm-start whenever the training pool changed (new stage size or
        # different split), or always on stage 0 to seed the head from random init.
        if (
            warm_start_fn is not None
            and ws_steps_stage > 0
            and (ws_pool_changed or stage_idx == 0)
        ):
            ws_kwargs_stage = dict(warm_start_kwargs)
            # Vary the warm-start seed per stage so the random pre-play sequence
            # (and the global RNG state warm-start ends in) is decorrelated
            # across stages -- otherwise every stage's "Iter 0" sees the same
            # secrets and the per-stage progress numbers are not independent.
            ws_kwargs_stage["seed"] = base_warm_seed + warm_start_seed_stride * stage_idx
            ws = warm_start_fn(
                policy,
                env,
                n_steps=ws_steps_stage,
                **ws_kwargs_stage,
            )
            ran_warm_start = True
            if verbose and isinstance(ws, dict):
                fitted = len(ws.get("loss", []))
                skipped = ws.get("skipped", "?")
                opt_steps = ws.get("opt_steps", "?")
                print(
                    f"Warm-start (stage {stage_idx + 1}, ws_pool={len(ws_pool)}): "
                    f"fitted {fitted} loss values across {opt_steps} opt steps; "
                    f"skipped {skipped}"
                )

        # Post-warm-start eval: in-distribution (training pool, env) and held-out
        # (env_eval, when configured). The gap between them is the memorization
        # signal; the gap between held-out post-WS and held-out end-of-stage ES
        # is the "is ES doing anything?" signal.
        post_ws_indist = float("nan")
        post_ws_heldout = float("nan")
        if quick_eval_success is not None and post_warm_start_eval_episodes > 0:
            post_ws_indist = float(
                quick_eval_success(
                    policy,
                    env,
                    n_episodes=post_warm_start_eval_episodes,
                    stochastic=not post_warm_start_eval_deterministic,
                    max_turns=es_kwargs.get("max_turns", 6),
                )
            )
            if env_eval is not None and eval_pool:
                post_ws_heldout = float(
                    quick_eval_success(
                        policy,
                        env_eval,
                        n_episodes=post_warm_start_eval_episodes,
                        stochastic=not post_warm_start_eval_deterministic,
                        max_turns=es_kwargs.get("max_turns", 6),
                    )
                )
        # Headline post_warmstart_success: held-out when available, else in-dist
        # (preserves the legacy semantics for callers that only inspect this key).
        post_ws_headline = (
            post_ws_heldout if (env_eval is not None and eval_pool and post_ws_heldout == post_ws_heldout)
            else post_ws_indist
        )
        combined["post_warmstart_success"].append(post_ws_headline)
        combined["post_warmstart_success_heldout"].append(post_ws_heldout)
        combined["post_warmstart_success_indist"].append(post_ws_indist)
        combined["stage_post_warmstart_iter"].append(iter_offset)

        # Trip the headroom ceiling on in-distribution post-WS success. We use
        # in-dist (not held-out) because that's the metric warm-start is
        # actually optimizing -- a saturated in-dist value means CE has fit
        # the training pool and additional warm-start steps would just refit
        # the same labels. Once tripped, all subsequent stages skip warm-start.
        if (
            warm_start_max_post_ws_success is not None
            and not warm_start_ceiling_tripped
            and post_ws_indist == post_ws_indist  # not NaN
            and post_ws_indist >= warm_start_max_post_ws_success
        ):
            warm_start_ceiling_tripped = True
            if verbose:
                print(
                    f"  [ceiling] post-WS in-dist {post_ws_indist:.1%} >= "
                    f"{warm_start_max_post_ws_success:.0%}; suppressing warm-start "
                    f"for stages {stage_idx + 2}-{len(vocab_schedule)} to leave ES "
                    f"headroom on harder pools."
                )
        if verbose:
            tag = "post-warm-start" if ran_warm_start else "post-expand (no warm-start)"
            mode = "greedy" if post_warm_start_eval_deterministic else "stoch"
            if env_eval is not None and eval_pool:
                hd_label = "fresh-eps" if holdout_mode == "episode" else "held-out"
                print(
                    f"  {tag} eval_success ({mode}, {post_warm_start_eval_episodes} eps): "
                    f"in-dist {post_ws_indist:.1%} | {hd_label} {post_ws_heldout:.1%}"
                )
            else:
                print(
                    f"  {tag} eval_success ({mode}, "
                    f"{post_warm_start_eval_episodes} eps): {post_ws_indist:.1%}"
                )

        # Optional masked-eval sanity probe: greedy eval on env_eval but with the head's
        # logits restricted to the in-vocab eval-pool indices. Diagnoses the prior 0% by
        # quantifying how much argmax mass is leaking outside the trained vocabulary.
        # See `holdout_mode` doc for the two semantic interpretations (word vs episode).
        post_ws_masked = float("nan")
        if (
            masked_eval_sanity_probe
            and quick_eval_success_masked is not None
            and env_eval is not None
            and eval_pool
            and post_warm_start_eval_episodes > 0
        ):
            allowed = {
                policy.word_to_idx[w]
                for w in eval_pool
                if w in getattr(policy, "word_to_idx", {})
            }
            if allowed:
                post_ws_masked = float(
                    quick_eval_success_masked(
                        policy,
                        env_eval,
                        allowed_indices=allowed,
                        n_episodes=post_warm_start_eval_episodes,
                        stochastic=not post_warm_start_eval_deterministic,
                        max_turns=es_kwargs.get("max_turns", 6),
                    )
                )
                if verbose:
                    print(
                        f"  [sanity] masked-greedy eval (argmax restricted to {len(allowed)} "
                        f"in-vocab indices, {post_warm_start_eval_episodes} eps): "
                        f"{post_ws_masked:.1%}  "
                        f"(unmasked held-out was {post_ws_heldout:.1%}; gap quantifies "
                        f"argmax leakage outside the eval-pool)"
                    )
        combined["stage_holdout_mode"].append(holdout_mode)
        combined["stage_masked_post_warmstart_success"].append(post_ws_masked)

        stage_history = train_es_wordle(
            policy=policy,
            env=env,
            n_iterations=per_stage_iters[stage_idx],
            verbose=verbose,
            env_eval=env_eval,
            restore_best_at_stage_end=restore_best_at_stage_end,
            eval_stochastic_every=eval_stochastic_every,
            **es_kwargs,
        )

        # Restore best-by-greedy-eval iterate (when tracking was active). This
        # decouples "did ES find a good iterate during this stage?" from "did
        # ES converge?" -- important under rotating mini-batch objectives
        # where ALPHA calibrated for a stationary objective can overshoot and
        # walk away from a strong mid-run iterate. Popping ``best_state`` off
        # the history dict avoids leaking a full CPU model copy into
        # ``combined`` (which callers often pickle).
        best_state_dict = stage_history.pop("best_state", None)
        if (
            restore_best_at_stage_end
            and stage_history.get("best_iter")
            and best_state_dict is not None
        ):
            stage_best_iter = int(stage_history["best_iter"][0])
            stage_best_succ = float(stage_history["best_eval_success"][0])
            policy.load_state_dict(best_state_dict, strict=False)
            if verbose:
                print(
                    f"  [restore-best] stage {stage_idx + 1}: restored iter "
                    f"{stage_best_iter} (eval_success={stage_best_succ:.1%})",
                    flush=True,
                )
            combined["best_iter"].append(stage_best_iter + iter_offset)
            combined["best_eval_success"].append(stage_best_succ)

        prev_ws_pool = ws_pool

        for k in _PER_ITER_KEYS:
            if k == "train_iter":
                combined[k].extend(int(i) + iter_offset for i in stage_history[k])
            else:
                combined[k].extend(stage_history[k])
        for k in _PER_EVAL_KEYS:
            if k == "iteration":
                combined[k].extend(int(i) + iter_offset for i in stage_history[k])
            else:
                # ``eval_success_stochastic`` is empty when
                # ``eval_stochastic_every is None`` (legacy path) -- safe extend.
                combined[k].extend(stage_history.get(k, []))

        combined["stage_starts"].append(iter_offset)
        combined["stage_vocab_sizes"].append(new_n)
        combined["stage_secret_pool_sizes"].append(len(ws_pool))
        combined["stage_eval_pool_sizes"].append(len(eval_pool))
        iter_offset += per_stage_iters[stage_idx]

    return combined
