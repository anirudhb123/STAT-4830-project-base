"""
Training script for Wordle environment with PPO.

This script trains a policy to play Wordle using Prime Intellect's
verifiers environment and PPO optimization.

Usage:
    python src/train_wordle.py --iterations 100 --episodes-per-iter 10
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from wordle_env import WordleEnvironmentWrapper, load_wordle_environment
from wordle_policy import WordleDiscretePolicy, WordleValueNetwork
from ppo_training import train_ppo_wordle, evaluate_policy


def main():
    parser = argparse.ArgumentParser(description='Train Wordle agent with PPO')
    
    # Environment args
    parser.add_argument('--num-train-episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--num-eval-episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--use-prime', action='store_true',
                        help='Use Prime Intellect backend (requires verifiers installed)')
    
    # Training args
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--episodes-per-iter', type=int, default=10,
                        help='Episodes to collect per iteration')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='PPO epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Minibatch size')
    parser.add_argument('--lr-policy', type=float, default=3e-4,
                        help='Policy learning rate')
    parser.add_argument('--lr-value', type=float, default=1e-3,
                        help='Value learning rate')
    parser.add_argument('--entropy-coef', type=float, default=0.02,
                        help='Entropy coefficient (higher = more exploration)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    
    # Model args
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for networks')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of layers in networks')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--log-wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Evaluate every N iterations')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize environment
    print("Initializing Wordle environment...")
    env = load_wordle_environment(
        num_train_examples=args.num_train_episodes,
        num_eval_examples=args.num_eval_episodes,
        use_prime_intellect=args.use_prime
    )
    
    # Initialize policy and value networks
    print("Initializing policy and value networks...")
    state_dim = 64  # Fixed embedding size from WordleEnvironmentWrapper
    
    policy = WordleDiscretePolicy(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)
    
    value_net = WordleValueNetwork(
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)
    
    print(f"Policy parameters: {policy.count_parameters():,}")
    print(f"Value network parameters: {value_net.count_parameters():,}")
    print(f"Vocabulary size: {len(policy.vocab)} words")
    
    # Initialize wandb if requested
    if args.log_wandb:
        try:
            import wandb
            wandb.init(
                project="wordle-ppo",
                config={
                    'iterations': args.iterations,
                    'episodes_per_iter': args.episodes_per_iter,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                    'lr_policy': args.lr_policy,
                    'lr_value': args.lr_value,
                    'entropy_coef': args.entropy_coef,
                    'clip_epsilon': args.clip_epsilon,
                    'gamma': args.gamma,
                    'gae_lambda': args.gae_lambda,
                    'seed': args.seed
                }
            )
        except ImportError:
            print("WARNING: wandb not installed. Install with: pip install wandb")
            args.log_wandb = False
    
    # Train
    print("\nStarting PPO training on Wordle...")
    print("=" * 60)
    
    trained_policy, trained_value, history = train_ppo_wordle(
        policy=policy,
        value_net=value_net,
        env=env,
        n_iterations=args.iterations,
        n_episodes_per_iter=args.episodes_per_iter,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        entropy_coef=args.entropy_coef,
        max_grad_norm=0.5,
        eval_every=args.eval_every,
        log_wandb=args.log_wandb,
        seed=args.seed
    )
    
    print("=" * 60)
    print("Training complete!")
    
    # Save models
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    torch.save(trained_policy.state_dict(), save_dir / 'wordle_policy.pt')
    torch.save(trained_value.state_dict(), save_dir / 'wordle_value.pt')
    print(f"Models saved to {save_dir}/")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_reward, eval_success, eval_turns = evaluate_policy(
        trained_policy, env, n_episodes=50, max_steps=6
    )
    print(f"Final Evaluation (50 episodes):")
    print(f"  Average Reward: {eval_reward:.3f}")
    print(f"  Success Rate: {eval_success:.2%}")
    print(f"  Average Turns: {eval_turns:.1f}")
    
    if args.log_wandb:
        wandb.log({
            'final/reward': eval_reward,
            'final/success_rate': eval_success,
            'final/avg_turns': eval_turns
        })
        wandb.finish()


if __name__ == '__main__':
    main()
