"""
Evolution Strategies training for Wordle environment.

Adapted from the GridWorld ES implementation to work with Wordle.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict


def _set_flat_params(policy, flat_params: torch.Tensor):
    """Set policy parameters from a flattened parameter vector."""
    offset = 0
    for param in policy.parameters():
        param_length = param.numel()
        param.data = flat_params[offset:offset + param_length].view_as(param).clone()
        offset += param_length


def es_gradient_estimate_wordle(
    policy,
    env,
    N: int = 20,
    sigma: float = 0.05,
    n_eval_episodes: int = 3,
    max_turns: int = 6
) -> Tuple[torch.Tensor, float, List[float]]:
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
        
        # Evaluate fitness (average reward over episodes)
        fitness = 0.0
        for _ in range(n_eval_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            turns = 0
            
            policy.eval()
            with torch.no_grad():
                while not done and turns < max_turns:
                    # Get state embedding
                    state_embedding = env.get_state_embedding(state)
                    
                    # Get action from policy
                    if hasattr(policy, 'format_action_xml'):
                        action_xml, _ = policy.format_action_xml(
                            state, state_embedding, deterministic=False
                        )
                    else:
                        action_idx, _ = policy.get_action(state_embedding, deterministic=False)
                        word = policy.vocab.action_to_word(action_idx)
                        action_xml = f"<guess>{word}</guess>"
                    
                    # Take step
                    state, reward, done, info = env.step(action_xml)
                    episode_reward += reward
                    turns += 1
            
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
    verbose: bool = True
) -> Dict[str, List]:
    """
    Train policy using Evolution Strategies on Wordle.
    
    Args:
        policy: WordleDiscretePolicy to train
        env: WordleEnvironmentWrapper
        N: Population size
        sigma: Noise scale
        alpha: Learning rate
        n_iterations: Number of training iterations
        n_eval_episodes: Episodes per fitness evaluation
        max_turns: Max turns per episode (6 for Wordle)
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
        'eval_turns': [],
        'gradient_norm': []
    }
    
    for iteration in range(n_iterations):
        # ES gradient step
        gradient, avg_fitness, fitness_values = es_gradient_estimate_wordle(
            policy, env,
            N=N,
            sigma=sigma,
            n_eval_episodes=n_eval_episodes,
            max_turns=max_turns
        )
        
        # Update parameters
        params = params + alpha * gradient
        _set_flat_params(policy, params)
        
        # Logging
        grad_norm = gradient.norm().item()
        
        # Periodic evaluation
        if iteration % eval_every == 0 or iteration == n_iterations - 1:
            # Evaluate on environment
            eval_rewards = []
            eval_successes = []
            eval_turn_counts = []
            
            policy.eval()
            with torch.no_grad():
                for _ in range(20):  # Eval on 20 episodes
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    turns = 0
                    
                    while not done and turns < max_turns:
                        state_embedding = env.get_state_embedding(state)
                        
                        if hasattr(policy, 'format_action_xml'):
                            action_xml, _ = policy.format_action_xml(
                                state, state_embedding, deterministic=True
                            )
                        else:
                            action_idx, _ = policy.get_action(state_embedding, deterministic=True)
                            word = policy.vocab.action_to_word(action_idx)
                            action_xml = f"<guess>{word}</guess>"
                        
                        state, reward, done, info = env.step(action_xml)
                        episode_reward += reward
                        turns += 1
                    
                    eval_rewards.append(episode_reward)
                    eval_successes.append(float(info.get('correct_answer', 0.0)))
                    eval_turn_counts.append(turns)
            
            eval_reward = np.mean(eval_rewards)
            eval_success = np.mean(eval_successes)
            eval_turns = np.mean(eval_turn_counts)
            
            history['iteration'].append(iteration)
            history['avg_fitness'].append(avg_fitness)
            history['eval_reward'].append(eval_reward)
            history['eval_success'].append(eval_success)
            history['eval_turns'].append(eval_turns)
            history['gradient_norm'].append(grad_norm)
            
            if verbose:
                print(f"Iter {iteration:4d} | "
                      f"Fitness: {avg_fitness:6.3f} | "
                      f"Eval Reward: {eval_reward:6.3f} | "
                      f"Success: {eval_success:5.1%} | "
                      f"Turns: {eval_turns:4.1f} | "
                      f"Grad Norm: {grad_norm:.4f}")
    
    return history
