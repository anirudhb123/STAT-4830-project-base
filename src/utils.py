"""
Helper functions for ES training and evaluation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path


NOISE_TYPES = ('gaussian', 'cauchy', 'laplace')
PARAM_MODES = ('all', 'lora')


def sample_perturbation(
    n_params: int,
    noise_type: str = 'gaussian',
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Sample a perturbation vector from the specified distribution.

    All distributions are centered at 0 with unit scale so that
    the ``sigma`` parameter in ES controls the effective noise magnitude.

    Args:
        n_params: Dimensionality of the perturbation vector.
        noise_type: One of 'gaussian', 'cauchy', or 'laplace'.

    Returns:
        epsilon: Tensor of shape (n_params,)
    """
    if noise_type == 'gaussian':
        return torch.randn(n_params, device=device)
    elif noise_type == 'cauchy':
        loc = torch.tensor(0.0, device=device)
        scale = torch.tensor(1.0, device=device)
        return torch.distributions.Cauchy(loc, scale).sample((n_params,))
    elif noise_type == 'laplace':
        loc = torch.tensor(0.0, device=device)
        scale = torch.tensor(1.0, device=device)
        return torch.distributions.Laplace(loc, scale).sample((n_params,))
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Choose from {NOISE_TYPES}."
        )


def score_function(epsilon: torch.Tensor, noise_type: str = 'gaussian') -> torch.Tensor:
    """
    Compute the negative score function  -∇_ε log p(ε)  for the given noise
    distribution.  This is the correct weighting to use in the ES gradient
    estimator for each distribution type.

    ∇_θ J ≈ (1/Nσ) Σ F_i · score(ε_i)

    Args:
        epsilon: Perturbation tensor of any shape.
        noise_type: One of 'gaussian', 'cauchy', or 'laplace'.

    Returns:
        Tensor of the same shape as epsilon.
    """
    if noise_type == 'gaussian':
        # -∇ log N(0,I) = ε
        return epsilon
    elif noise_type == 'cauchy':
        # -∇ log Cauchy(0,1) = 2ε / (1 + ε²)
        return 2 * epsilon / (1 + epsilon ** 2)
    elif noise_type == 'laplace':
        # -∇ log Laplace(0,1) = sign(ε)
        return torch.sign(epsilon)
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Choose from {NOISE_TYPES}."
        )


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 20,
    max_steps: int = 100,
    deterministic: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluate policy on environment.
    
    Args:
        policy: PolicyNetwork
        env: GridWorld environment
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy
        
    Returns:
        avg_reward: Average episode reward
        success_rate: Fraction of successful episodes
        avg_steps: Average steps to goal
    """
    total_rewards = []
    successes = []
    episode_steps = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action, _ = policy.get_action(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        successes.append(float(info['success']))
        episode_steps.append(steps)
    
    return (
        np.mean(total_rewards),
        np.mean(successes),
        np.mean(episode_steps)
    )


def es_gradient_estimate(
    policy,
    env,
    N: int = 20,
    sigma: float = 0.05,
    n_eval_episodes: int = 5,
    max_steps: int = 50,
    noise_type: str = 'gaussian',
    param_mode: str = 'all',
    return_metadata: bool = False
) -> Union[
    Tuple[torch.Tensor, float, List[float]],
    Tuple[torch.Tensor, float, List[float], Dict[str, float]]
]:
    """
    Estimate gradient using Evolution Strategies.
    
    Algorithm:
        1. Sample N perturbations ε_i from the chosen distribution
        2. Evaluate fitness R(θ + σε_i) for each perturbation
        3. Estimate gradient: ∇J ≈ (1/Nσ) Σ R(θ + σε_i) · ε_i
    
    Args:
        policy: PolicyNetwork to optimize
        env: Environment for evaluation
        N: Population size (number of perturbations)
        sigma: Noise scale
        n_eval_episodes: Episodes per perturbation evaluation
        max_steps: Max steps per episode
        noise_type: Perturbation distribution — 'gaussian', 'cauchy', or 'laplace'
        param_mode: Parameter subset to optimize — 'all' or 'lora'
        return_metadata: If True, return extra diagnostics
    
    Returns:
        gradient: Estimated gradient (flattened parameter vector)
        avg_fitness: Average fitness across population
        fitness_values: List of fitness values for all perturbations
        metadata: Optional dict with keys:
            - fitness_std
            - env_interactions
    """
    # Get flattened parameters
    target_params = _select_params(policy, param_mode)
    params = _flatten_params(target_params)
    policy_device = next(policy.parameters()).device
    n_params = params.shape[0]
    if n_params == 0:
        raise ValueError(f"No parameters selected with param_mode='{param_mode}'.")
    
    # Storage
    perturbations = []
    fitness_values = []
    total_env_interactions = 0
    
    # Sample and evaluate perturbations
    for i in range(N):
        # Sample perturbation from chosen distribution
        epsilon = sample_perturbation(n_params, noise_type, device=policy_device)
        perturbations.append(epsilon)
        
        # Perturb parameters
        perturbed_params = params + sigma * epsilon
        
        # Set perturbed parameters
        _set_selected_params(target_params, perturbed_params)
        
        # Evaluate fitness
        fitness = 0.0
        for _ in range(n_eval_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                action, _ = policy.get_action(state, deterministic=False)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                steps += 1
                total_env_interactions += 1
            
            fitness += episode_reward
        
        fitness /= n_eval_episodes
        fitness_values.append(fitness)
    
    # Restore original parameters
    _set_selected_params(target_params, params)
    
    # Compute gradient estimate
    fitness_tensor = torch.tensor(
        fitness_values,
        dtype=torch.float32,
        device=policy_device
    )
    perturbations_tensor = torch.stack(perturbations)
    
    # Apply the correct score function for the chosen noise distribution
    score_weights = score_function(perturbations_tensor, noise_type)
    
    # Standardize fitness (improves stability)
    fitness_std = fitness_tensor.std()
    if fitness_std > 1e-8:
        fitness_normalized = (fitness_tensor - fitness_tensor.mean()) / fitness_std
    else:
        fitness_normalized = fitness_tensor - fitness_tensor.mean()
    
    # Gradient estimate: (1/Nσ) Σ F_i · score(ε_i)
    gradient = (score_weights.T @ fitness_normalized) / (N * sigma)

    if return_metadata:
        metadata = {
            'fitness_std': float(fitness_std.item()),
            'env_interactions': float(total_env_interactions)
        }
        return gradient, fitness_tensor.mean().item(), fitness_values, metadata

    return gradient, fitness_tensor.mean().item(), fitness_values


def train_es(
    policy,
    env,
    N: int = 20,
    sigma: float = 0.05,
    alpha: float = 0.01,
    n_iterations: int = 100,
    n_eval_episodes: int = 5,
    max_steps: int = 50,
    eval_every: int = 10,
    verbosity: int = 0,
    noise_type: str = 'gaussian',
    param_mode: str = 'all'
) -> Dict[str, List]:
    """
    Train policy using Evolution Strategies.
    
    Args:
        policy: PolicyNetwork to train
        env: Environment
        N: Population size
        sigma: Noise scale
        alpha: Learning rate
        n_iterations: Number of training iterations
        n_eval_episodes: Episodes per fitness evaluation
        max_steps: Max steps per episode
        eval_every: Evaluate policy every N iterations
        verbosity: Print progress every verbosity iterations (0 means no progress printing)
        noise_type: Perturbation distribution — 'gaussian', 'cauchy', or 'laplace'
        param_mode: Parameter subset to optimize — 'all' or 'lora'
    
    Returns:
        history: Dictionary with training history
    """
    # Get flattened parameters
    target_params = _select_params(policy, param_mode)
    params = _flatten_params(target_params)
    
    # Training history
    history = {
        'iteration': [],
        'avg_fitness': [],
        'eval_reward': [],
        'eval_success': [],
        'eval_steps': [],
        'gradient_norm': [],
        'fitness_std': [],
        'env_interactions': [],
        'cumulative_env_interactions': []
    }

    cumulative_env_interactions = 0.0
    
    for iteration in range(n_iterations):
        # ES gradient step
        gradient, avg_fitness, fitness_values, grad_meta = es_gradient_estimate(
            policy, env,
            N=N,
            sigma=sigma,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps,
            noise_type=noise_type,
            param_mode=param_mode,
            return_metadata=True
        )
        
        # Update parameters
        params = params + alpha * gradient
        _set_selected_params(target_params, params)
        
        # Logging
        grad_norm = gradient.norm().item()
        fitness_std = grad_meta['fitness_std']
        cumulative_env_interactions += grad_meta['env_interactions']
        
        # Periodic evaluation
        if iteration % eval_every == 0 or iteration == n_iterations - 1:
            eval_reward, eval_success, eval_steps = evaluate_policy(
                policy, env,
                n_episodes=20,
                max_steps=max_steps,
                deterministic=True
            )
            
            history['iteration'].append(iteration)
            history['avg_fitness'].append(avg_fitness)
            history['eval_reward'].append(eval_reward)
            history['eval_success'].append(eval_success)
            history['eval_steps'].append(eval_steps)
            history['gradient_norm'].append(grad_norm)
            history['fitness_std'].append(fitness_std)
            history['env_interactions'].append(grad_meta['env_interactions'])
            history['cumulative_env_interactions'].append(cumulative_env_interactions)
            
            if verbosity > 0 and (iteration % verbosity == 0 or iteration == n_iterations - 1):
                print(f"Iter {iteration:4d}/{n_iterations:4d} | "
                      f"Fitness: {avg_fitness:6.3f} | "
                      f"Eval Reward: {eval_reward:6.3f} | "
                      f"Success: {eval_success:.2%} | "
                      f"Grad Norm: {grad_norm:.4f} | "
                      f"Fit Std: {fitness_std:.3f}")
    
    return history


def plot_training_curves(
    history: Dict[str, List],
    save_path: Optional[Path] = None
):
    """
    Plot training curves from history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = history['iteration']
    
    # Reward
    axes[0, 0].plot(iterations, history['eval_reward'], linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Evaluation Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate
    axes[0, 1].plot(iterations, history['eval_success'], linewidth=2, color='green')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Evaluation Success Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fitness
    axes[1, 0].plot(iterations, history['avg_fitness'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Average Fitness')
    axes[1, 0].set_title('Training Fitness')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1, 1].plot(iterations, history['gradient_norm'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Gradient Magnitude (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _set_flat_params(model: torch.nn.Module, flat_params: torch.Tensor):
    """
    Set model parameters from flattened parameter vector.
    
    Args:
        model: PyTorch model
        flat_params: Flattened parameter tensor
    """
    _set_selected_params(list(model.parameters()), flat_params)


def _flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten a parameter list into one vector."""
    if len(params) == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([p.flatten() for p in params])


def _set_selected_params(params: List[torch.nn.Parameter], flat_params: torch.Tensor):
    """Set a selected parameter list from a flattened tensor."""
    offset = 0
    for param in params:
        numel = param.numel()
        flat_slice = flat_params[offset:offset+numel]
        param.data = flat_slice.to(device=param.device, dtype=param.dtype).view_as(param).clone()
        offset += numel


def _select_params(policy, param_mode: str = 'all') -> List[torch.nn.Parameter]:
    """Select a parameter subset from policy according to param_mode."""
    if param_mode not in PARAM_MODES:
        raise ValueError(f"Unknown param_mode '{param_mode}'. Choose from {PARAM_MODES}.")

    if param_mode == 'all':
        return list(policy.parameters())

    if not hasattr(policy, 'lora_parameters'):
        raise ValueError("Policy does not expose lora_parameters() required for param_mode='lora'.")

    return list(policy.lora_parameters())


def compute_statistics(
    results: Dict[str, Dict[str, List]],
    metrics: List[str] = ['rewards', 'successes', 'steps']
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compute mean and std for results across trials.
    
    Args:
        results: Nested dict with structure results[method][metric] = [trial_values]
        metrics: List of metric names to compute statistics for
    
    Returns:
        stats: Dict with structure stats[method][metric] = (mean, std)
    """
    stats = {}
    
    for method in results.keys():
        stats[method] = {}
        for metric in metrics:
            values = results[method][metric]
            stats[method][metric] = (np.mean(values), np.std(values))
    
    return stats


def print_comparison_table(
    stats: Dict[str, Dict[str, Tuple[float, float]]],
    methods: List[str] = ['random', 'es', 'ppo']
):
    """
    Print formatted comparison table.
    
    Args:
        stats: Statistics dict from compute_statistics
        methods: List of method names to include
    """
    print("\n" + "="*80)
    print("METHOD COMPARISON (mean ± std)")
    print("="*80)
    print(f"{'Method':<10} {'Reward':<20} {'Success Rate':<20} {'Steps':<20}")
    print("-"*80)
    
    for method in methods:
        if method not in stats:
            continue
        
        reward_mean, reward_std = stats[method]['rewards']
        success_mean, success_std = stats[method]['successes']
        steps_mean, steps_std = stats[method]['steps']
        
        print(f"{method.upper():<10} "
              f"{reward_mean:6.3f} ± {reward_std:5.3f}     "
              f"{success_mean:5.3f} ± {success_std:5.3f}      "
              f"{steps_mean:5.1f} ± {steps_std:4.1f}")
    
    print("="*80 + "\n")
