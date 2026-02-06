"""
Compare ES, PPO, and Random baseline on GridWorld.
This is the main experiment script for the 1-week proof of life.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from gridworld_env import GridWorld, HarderGridWorld
from policy_network import PolicyNetwork, ValueNetwork
from train_es_gridworld import train_es, evaluate_policy as eval_es
from train_ppo_gridworld import train_ppo, evaluate_policy as eval_ppo


class RandomPolicy:
    """Random baseline policy."""
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
    
    def get_action(self, state, deterministic=False):
        return np.random.randint(0, self.n_actions), None


def evaluate_random(env, n_episodes: int = 20, max_steps: int = 100):
    """Evaluate random policy."""
    policy = RandomPolicy(env.n_actions)
    total_rewards = []
    successes = []
    episode_steps = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action, _ = policy.get_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        successes.append(float(info['success']))
        episode_steps.append(steps)
    
    return np.mean(total_rewards), np.mean(successes), np.mean(episode_steps)


def run_comparison(
    env_class,
    env_kwargs: dict,
    n_trials: int = 5,
    n_iterations: int = 100,
    output_dir: str = "./results"
):
    """
    Run comparison experiment across multiple trials.
    
    Returns dict with results for each method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get environment dimensions
    dummy_env = env_class(**env_kwargs)
    state_dim = dummy_env._get_state().shape[0]
    action_dim = dummy_env.n_actions
    max_steps = env_kwargs["max_steps"]
    
    print(f"=" * 80)
    print(f"Running comparison on {env_class.__name__}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print(f"=" * 80)
    
    results = {
        "random": {"rewards": [], "successes": [], "steps": []},
        "es": {"rewards": [], "successes": [], "steps": []},
        "ppo": {"rewards": [], "successes": [], "steps": []},
    }
    
    for trial in range(n_trials):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print(f"{'='*80}")
        seed = 42 + trial
        
        # 1. Random baseline
        print("\n[1/3] Evaluating random policy...")
        env = env_class(**env_kwargs)
        rand_reward, rand_success, rand_steps = evaluate_random(
            env, n_episodes=50, max_steps=max_steps
        )
        results["random"]["rewards"].append(rand_reward)
        results["random"]["successes"].append(rand_success)
        results["random"]["steps"].append(rand_steps)
        print(f"Random: reward={rand_reward:.3f}, success={rand_success:.2f}, steps={rand_steps:.1f}")
        
        # 2. ES
        print("\n[2/3] Training ES policy...")
        es_policy = PolicyNetwork(state_dim, action_dim, hidden_dim=64, n_layers=2).to(device)
        es_policy = train_es(
            policy=es_policy,
            env_class=env_class,
            env_kwargs=env_kwargs,
            N=20,
            sigma=0.05,
            alpha=0.01,
            T=n_iterations,
            n_eval_episodes=5,
            max_steps=max_steps,
            eval_every=max(n_iterations // 10, 1),
            log_wandb=False,
            seed=seed
        )
        env = env_class(**env_kwargs)
        es_reward, es_success, es_steps = eval_es(
            es_policy, env, n_episodes=50, max_steps=max_steps
        )
        results["es"]["rewards"].append(es_reward)
        results["es"]["successes"].append(es_success)
        results["es"]["steps"].append(es_steps)
        print(f"ES: reward={es_reward:.3f}, success={es_success:.2f}, steps={es_steps:.1f}")
        
        # 3. PPO
        print("\n[3/3] Training PPO policy...")
        ppo_policy = PolicyNetwork(state_dim, action_dim, hidden_dim=64, n_layers=2).to(device)
        ppo_value = ValueNetwork(state_dim, hidden_dim=64, n_layers=2).to(device)
        ppo_policy, ppo_value = train_ppo(
            policy=ppo_policy,
            value_net=ppo_value,
            env_class=env_class,
            env_kwargs=env_kwargs,
            n_iterations=n_iterations,
            n_steps=128,
            n_epochs=4,
            batch_size=64,
            eval_every=max(n_iterations // 10, 1),
            log_wandb=False,
            seed=seed
        )
        env = env_class(**env_kwargs)
        ppo_reward, ppo_success, ppo_steps = eval_ppo(
            ppo_policy, env, n_episodes=50, max_steps=max_steps
        )
        results["ppo"]["rewards"].append(ppo_reward)
        results["ppo"]["successes"].append(ppo_success)
        results["ppo"]["steps"].append(ppo_steps)
        print(f"PPO: reward={ppo_reward:.3f}, success={ppo_success:.2f}, steps={ppo_steps:.1f}")
    
    # Compute statistics
    print(f"\n{'='*80}")
    print("FINAL RESULTS (mean ± std)")
    print(f"{'='*80}")
    
    for method in ["random", "es", "ppo"]:
        rewards = results[method]["rewards"]
        successes = results[method]["successes"]
        steps = results[method]["steps"]
        
        print(f"{method.upper():8s}: "
              f"reward={np.mean(rewards):.3f}±{np.std(rewards):.3f}, "
              f"success={np.mean(successes):.3f}±{np.std(successes):.3f}, "
              f"steps={np.mean(steps):.1f}±{np.std(steps):.1f}")
    
    # Plot results
    plot_results(results, output_path / f"comparison_{env_class.__name__}.png")
    
    return results


def plot_results(results: dict, save_path: Path):
    """Plot comparison results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ["random", "es", "ppo"]
    colors = ["gray", "blue", "orange"]
    
    metrics = [
        ("rewards", "Average Reward"),
        ("successes", "Success Rate"),
        ("steps", "Steps to Goal")
    ]
    
    for ax, (metric, label) in zip(axes, metrics):
        data = [results[m][metric] for m in methods]
        
        # Box plot
        bp = ax.boxplot(data, labels=[m.upper() for m in methods],
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Add individual points
        for i, (method_data, color) in enumerate(zip(data, colors)):
            x = np.random.normal(i + 1, 0.04, size=len(method_data))
            ax.scatter(x, method_data, alpha=0.6, color=color, s=50)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_xlabel("Method", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(label, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare ES vs PPO vs Random on GridWorld")
    parser.add_argument("--env", type=str, default="simple", choices=["simple", "harder"],
                       help="Environment type")
    parser.add_argument("--size", type=int, default=8, help="Grid size")
    parser.add_argument("--obstacles", type=int, default=8, help="Number of obstacles")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--iterations", type=int, default=100, help="Training iterations")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup environment
    env_class = HarderGridWorld if args.env == "harder" else GridWorld
    env_kwargs = {
        "size": args.size,
        "n_obstacles": args.obstacles,
        "max_steps": 50 if args.env == "simple" else 100,
        "seed": 42,
    }
    
    # Run comparison
    results = run_comparison(
        env_class=env_class,
        env_kwargs=env_kwargs,
        n_trials=args.trials,
        n_iterations=args.iterations,
        output_dir=args.output
    )
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
