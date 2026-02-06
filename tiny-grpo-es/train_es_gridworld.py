"""
Evolution Strategies (ES) training for GridWorld policy optimization.
Adapted from train_es.py for parameter-space optimization.
"""
import numpy as np
import torch
import random
from typing import List, Tuple
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    WANDB_AVAILABLE = False
    print(f"Warning: wandb not available ({type(e).__name__}). Install/reinstall with 'pip install --upgrade wandb' to enable logging.")
from pathlib import Path

from gridworld_env import GridWorld, HarderGridWorld
from policy_network import PolicyNetwork


def generate_noise(model: torch.nn.Module, seed: int) -> List[torch.Tensor]:
    """Generate and return noise vectors for all parameters."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_vectors = []
    for param in model.parameters():
        noise = torch.randn(
            param.shape,
            generator=generator,
            dtype=param.dtype,
            device=param.device
        )
        noise_vectors.append(noise)
    return noise_vectors


def apply_noise(model: torch.nn.Module, noise_vectors: List[torch.Tensor], scale: float):
    """Apply noise vectors to model parameters."""
    for param, noise in zip(model.parameters(), noise_vectors):
        param.data.add_(noise, alpha=scale)


def evaluate_policy(
    policy: PolicyNetwork,
    env: GridWorld,
    n_episodes: int = 5,
    max_steps: int = 100,
    deterministic: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluate policy on environment.
    
    Returns:
        mean_reward: average total reward
        success_rate: fraction of successful episodes
        mean_steps: average steps per episode
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
            # Get action from policy
            action, _ = policy.get_action(state, deterministic=deterministic)
            
            # Take step
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        successes.append(float(info['success']))
        episode_steps.append(steps)
    
    mean_reward = np.mean(total_rewards)
    success_rate = np.mean(successes)
    mean_steps = np.mean(episode_steps)
    
    return mean_reward, success_rate, mean_steps


def train_es(
    policy: PolicyNetwork,
    env_class,
    env_kwargs: dict,
    N: int = 20,
    sigma: float = 0.05,
    alpha: float = 0.01,
    T: int = 100,
    n_episodes: int = 5,
    max_steps: int = 100,
    eval_every: int = 5,
    log_wandb: bool = True,
    seed: int = 42
):
    """
    Train policy using Evolution Strategies.
    
    Algorithm:
        1. For each iteration:
            a. Generate N perturbations of policy parameters
            b. Evaluate each perturbed policy
            c. Compute gradient estimate using rewards
            d. Update policy parameters
    
    Args:
        policy: Policy network to optimize
        env_class: Environment class (e.g., GridWorld)
        env_kwargs: Kwargs for environment initialization
        N: Population size (number of perturbations per iteration)
        sigma: Noise scale for parameter perturbations
        alpha: Learning rate
        T: Number of iterations
        n_episodes: Episodes per evaluation
        max_steps: Max steps per episode
        eval_every: Evaluate every N iterations
        log_wandb: Whether to log to wandb
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = next(policy.parameters()).device
    
    # Initialize tracking
    best_reward = -float('inf')
    ema_reward = None
    
    print(f"Starting ES training for {T} iterations...")
    print(f"Policy parameters: {policy.count_parameters()}")
    print(f"N={N}, sigma={sigma}, alpha={alpha}")
    
    for t in range(T):
        # Create fresh environment for this iteration
        env = env_class(**env_kwargs)
        
        # Generate random seeds for reproducibility
        seeds = [random.randint(0, 2**31 - 1) for _ in range(N)]
        
        # Evaluate perturbations
        rewards = []
        for n in range(N):
            # Generate and apply noise
            noise_vectors = generate_noise(policy, seeds[n])
            apply_noise(policy, noise_vectors, sigma)
            
            # Evaluate perturbed policy
            reward, success_rate, steps = evaluate_policy(
                policy, env, n_episodes, max_steps, deterministic=True
            )
            rewards.append(reward)
            
            # Restore policy
            apply_noise(policy, noise_vectors, -sigma)
        
        # Normalize rewards (z-score)
        rewards = np.array(rewards)
        R_mean = rewards.mean()
        R_std = rewards.std()
        if R_std > 1e-8:
            z_scores = (rewards - R_mean) / R_std
        else:
            z_scores = np.zeros_like(rewards)
        
        # Compute gradient estimate
        gradient_accum = [torch.zeros_like(p) for p in policy.parameters()]
        
        for n in range(N):
            noise_vectors = generate_noise(policy, seeds[n])
            z_score = z_scores[n]
            
            for grad, noise in zip(gradient_accum, noise_vectors):
                # ES gradient: E[R(θ + σε) * ε] ≈ (1/N) Σ z_i * ε_i
                grad.add_(noise, alpha=z_score / N)
        
        # Apply gradient
        for param, grad in zip(policy.parameters(), gradient_accum):
            param.data.add_(grad, alpha=alpha)
        
        # Update EMA
        if ema_reward is None:
            ema_reward = R_mean
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * R_mean
        
        # Track best
        if R_mean > best_reward:
            best_reward = R_mean
        
        # Logging
        if (t + 1) % eval_every == 0 or t == 0:
            # Evaluate current policy
            eval_reward, eval_success, eval_steps = evaluate_policy(
                policy, env, n_episodes=10, max_steps=max_steps
            )
            
            print(f"Iter {t+1}/{T}: reward_mean={R_mean:.3f}, reward_std={R_std:.3f}, "
                  f"eval_reward={eval_reward:.3f}, eval_success={eval_success:.2f}, "
                  f"eval_steps={eval_steps:.1f}, ema={ema_reward:.3f}, best={best_reward:.3f}")
            
            if log_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "iteration": t + 1,
                    "reward_mean": R_mean,
                    "reward_std": R_std,
                    "reward_max": rewards.max(),
                    "reward_min": rewards.min(),
                    "reward_ema": ema_reward,
                    "eval_reward": eval_reward,
                    "eval_success_rate": eval_success,
                    "eval_steps": eval_steps,
                    "best_reward": best_reward,
                }, step=t + 1)
    
    print(f"\nTraining complete! Best reward: {best_reward:.3f}")
    return policy


def main():
    # Hyperparameters
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Environment settings
    use_harder_env = False  # Set to True for key-door environment
    env_class = HarderGridWorld if use_harder_env else GridWorld
    env_kwargs = {
        "size": 8,
        "n_obstacles": 8,
        "max_steps": 50,
        "seed": seed,
    }
    if use_harder_env:
        env_kwargs["max_steps"] = 100
    
    # Create dummy env to get dimensions
    dummy_env = env_class(**env_kwargs)
    state_dim = dummy_env._get_state().shape[0]
    action_dim = dummy_env.n_actions
    
    # Policy network
    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        n_layers=2
    ).to(device)
    
    # ES hyperparameters
    N = 30  # population size
    sigma = 0.05  # noise scale
    alpha = 0.01  # learning rate
    T = 200  # iterations
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="stat-4830-es-gridworld",
            name=f"ES-grid{env_kwargs['size']}-harder{use_harder_env}",
            config={
                "seed": seed,
                "env": env_class.__name__,
                "env_kwargs": env_kwargs,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "policy_params": policy.count_parameters(),
                "N": N,
                "sigma": sigma,
                "alpha": alpha,
                "T": T,
                "method": "ES",
            }
        )
    
    # Train
    policy = train_es(
        policy=policy,
        env_class=env_class,
        env_kwargs=env_kwargs,
        N=N,
        sigma=sigma,
        alpha=alpha,
        T=T,
        n_episodes=5,
        max_steps=env_kwargs["max_steps"],
        eval_every=5,
        log_wandb=True,
        seed=seed,
    )
    
    # Save policy
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / f"es_policy_{env_class.__name__}.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Policy saved to {save_path}")
    
    # Final evaluation
    print("\nFinal evaluation (20 episodes):")
    final_env = env_class(**env_kwargs)
    final_reward, final_success, final_steps = evaluate_policy(
        policy, final_env, n_episodes=20, max_steps=env_kwargs["max_steps"]
    )
    print(f"Reward: {final_reward:.3f}, Success: {final_success:.2f}, Steps: {final_steps:.1f}")
    
    if WANDB_AVAILABLE:
        wandb.log({
            "final/reward": final_reward,
            "final/success_rate": final_success,
            "final/steps": final_steps,
        })
        
        wandb.finish()


if __name__ == "__main__":
    main()
