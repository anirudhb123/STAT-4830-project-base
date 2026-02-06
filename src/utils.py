"""
Helper functions for ES training and evaluation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from pathlib import Path


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
    max_steps: int = 50
) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Estimate gradient using Evolution Strategies.
    
    Algorithm:
        1. Sample N perturbations ε_i ~ N(0, I)
        2. Evaluate fitness R(θ + σε_i) for each perturbation
        3. Estimate gradient: ∇J ≈ (1/Nσ) Σ R(θ + σε_i) · ε_i
    
    Args:
        policy: PolicyNetwork to optimize
        env: Environment for evaluation
        N: Population size (number of perturbations)
        sigma: Noise scale
        n_eval_episodes: Episodes per perturbation evaluation
        max_steps: Max steps per episode
    
    Returns:
        gradient: Estimated gradient (flattened parameter vector)
        avg_fitness: Average fitness across population
        fitness_values: List of fitness values for all perturbations
    """
    # Get flattened parameters
    params = torch.cat([p.flatten() for p in policy.parameters()])
    n_params = params.shape[0]
    
    # Storage
    perturbations = []
    fitness_values = []
    
    # Sample and evaluate perturbations
    for i in range(N):
        # Sample perturbation
        epsilon = torch.randn(n_params)
        perturbations.append(epsilon)
        
        # Perturb parameters
        perturbed_params = params + sigma * epsilon
        
        # Set perturbed parameters
        _set_flat_params(policy, perturbed_params)
        
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
            
            fitness += episode_reward
        
        fitness /= n_eval_episodes
        fitness_values.append(fitness)
    
    # Restore original parameters
    _set_flat_params(policy, params)
    
    # Compute gradient estimate
    fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32)
    perturbations_tensor = torch.stack(perturbations)
    
    # Standardize fitness (improves stability)
    fitness_std = fitness_tensor.std()
    if fitness_std > 1e-8:
        fitness_normalized = (fitness_tensor - fitness_tensor.mean()) / fitness_std
    else:
        fitness_normalized = fitness_tensor - fitness_tensor.mean()
    
    # Gradient estimate: (1/Nσ) Σ F_i · ε_i
    gradient = (perturbations_tensor.T @ fitness_normalized) / (N * sigma)
    
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
    verbose: bool = True
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
        verbose: Print progress
    
    Returns:
        history: Dictionary with training history
    """
    # Get flattened parameters
    params = torch.cat([p.flatten() for p in policy.parameters()])
    
    # Training history
    history = {
        'iteration': [],
        'avg_fitness': [],
        'eval_reward': [],
        'eval_success': [],
        'gradient_norm': []
    }
    
    for iteration in range(n_iterations):
        # ES gradient step
        gradient, avg_fitness, fitness_values = es_gradient_estimate(
            policy, env,
            N=N,
            sigma=sigma,
            n_eval_episodes=n_eval_episodes,
            max_steps=max_steps
        )
        
        # Update parameters
        params = params + alpha * gradient
        _set_flat_params(policy, params)
        
        # Logging
        grad_norm = gradient.norm().item()
        
        # Periodic evaluation
        if iteration % eval_every == 0 or iteration == n_iterations - 1:
            eval_reward, eval_success, _ = evaluate_policy(
                policy, env,
                n_episodes=20,
                max_steps=max_steps,
                deterministic=True
            )
            
            history['iteration'].append(iteration)
            history['avg_fitness'].append(avg_fitness)
            history['eval_reward'].append(eval_reward)
            history['eval_success'].append(eval_success)
            history['gradient_norm'].append(grad_norm)
            
            if verbose:
                print(f"Iter {iteration:4d} | "
                      f"Fitness: {avg_fitness:6.3f} | "
                      f"Eval Reward: {eval_reward:6.3f} | "
                      f"Success: {eval_success:.2%} | "
                      f"Grad Norm: {grad_norm:.4f}")
    
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
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = flat_params[offset:offset+numel].view_as(param).clone()
        offset += numel


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
