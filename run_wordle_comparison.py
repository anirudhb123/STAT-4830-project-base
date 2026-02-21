#!/usr/bin/env python
"""
Full ES vs PPO comparison on Wordle (matching week4_implementation.ipynb structure).

This script runs both methods and generates comparison plots.
"""

import sys
import os
sys.path.append('src')

# Set NLTK data path
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), '.venv', 'nltk_data')

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from wordle_env import load_wordle_environment
from wordle_policy import WordleDiscretePolicy, WordleValueNetwork
from wordle_es import train_es_wordle
from ppo_training import train_ppo_wordle, evaluate_policy

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 4)


def main():
    print("=" * 70)
    print("WORDLE: ES vs PPO Comparison")
    print("=" * 70)
    
    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cpu')
    
    # Create environment
    print("\n[1/4] Loading Wordle environment...")
    env = load_wordle_environment(
        num_train_examples=2000,
        num_eval_examples=20,
        use_prime_intellect=True  # Use real Prime Intellect
    )
    print(f"✓ Environment ready")
    
    # Train ES
    print("\n[2/4] Training with Evolution Strategies...")
    print("-" * 70)
    policy_es = WordleDiscretePolicy(
        state_dim=64,
        hidden_dim=128,
        n_layers=3
    ).to(device)
    
    history_es = train_es_wordle(
        policy=policy_es,
        env=env,
        N=30,
        sigma=0.1,
        alpha=0.03,
        n_iterations=50,
        n_eval_episodes=3,
        max_turns=6,
        eval_every=5,
        verbose=True
    )
    print("✓ ES training complete")
    
    # Train PPO
    print("\n[3/4] Training with PPO...")
    print("-" * 70)
    policy_ppo = WordleDiscretePolicy(
        state_dim=64,
        hidden_dim=128,
        n_layers=3
    ).to(device)
    
    value_net = WordleValueNetwork(
        state_dim=64,
        hidden_dim=128,
        n_layers=3
    ).to(device)
    
    trained_policy_ppo, trained_value, history_ppo = train_ppo_wordle(
        policy=policy_ppo,
        value_net=value_net,
        env=env,
        n_iterations=50,
        n_episodes_per_iter=10,
        n_epochs=4,
        batch_size=32,
        lr_policy=3e-4,
        lr_value=1e-3,
        entropy_coef=0.02,
        eval_every=5,
        log_wandb=False,
        seed=42
    )
    print("✓ PPO training complete")
    
    # Final evaluation
    print("\n[4/4] Final Evaluation...")
    print("-" * 70)
    
    es_reward, es_success, es_turns = evaluate_policy(
        policy_es, env, n_episodes=50, max_steps=6
    )
    
    ppo_reward, ppo_success, ppo_turns = evaluate_policy(
        trained_policy_ppo, env, n_episodes=50, max_steps=6
    )
    
    print(f"\nEvolution Strategies:")
    print(f"  Success Rate: {es_success:.1%}")
    print(f"  Avg Reward: {es_reward:.3f}")
    print(f"  Avg Turns: {es_turns:.2f}")
    
    print(f"\nPPO:")
    print(f"  Success Rate: {ppo_success:.1%}")
    print(f"  Avg Reward: {ppo_reward:.3f}")
    print(f"  Avg Turns: {ppo_turns:.2f}")
    
    # Plot results
    print("\n[5/5] Generating plots...")
    os.makedirs('figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['ES', 'PPO']
    success_rates = [es_success, ppo_success]
    avg_rewards = [es_reward, ppo_reward]
    avg_turns = [es_turns, ppo_turns]
    
    # Success Rate
    axes[0].bar(methods, success_rates, color=['blue', 'orange'], alpha=0.7)
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
    
    # Average Reward
    axes[1].bar(methods, avg_rewards, color=['blue', 'orange'], alpha=0.7)
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Reward Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(avg_rewards):
        axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    
    # Average Turns
    axes[2].bar(methods, avg_turns, color=['blue', 'orange'], alpha=0.7)
    axes[2].set_ylabel('Average Turns')
    axes[2].set_title('Efficiency Comparison')
    axes[2].set_ylim([1, 6])
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(avg_turns):
        axes[2].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figures/wordle_final_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved figures/wordle_final_comparison.png")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    torch.save(policy_es.state_dict(), 'models/wordle_es_final.pt')
    torch.save(trained_policy_ppo.state_dict(), 'models/wordle_ppo_final.pt')
    torch.save(trained_value.state_dict(), 'models/wordle_value_final.pt')
    print("✓ Saved models to models/")
    
    print("\n" + "=" * 70)
    print(f"Winner by success rate: {'ES' if es_success > ppo_success else 'PPO'}")
    print(f"Winner by efficiency: {'ES' if es_turns < ppo_turns else 'PPO'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
