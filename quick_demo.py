#!/usr/bin/env python3
"""
Quick demonstration of ES training on GridWorld.
Run this to verify everything works (~2 minutes).

Usage: python quick_demo.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from model import GridWorld, PolicyNetwork
from utils import train_es, evaluate_policy, plot_training_curves
from pathlib import Path

def main():
    print("="*80)
    print("QUICK DEMO: Evolution Strategies for GridWorld")
    print("="*80)
    
    # Setup
    print("\n[1/4] Setting up environment...")
    env = GridWorld(size=8, n_obstacles=8, max_steps=50, seed=42)
    policy = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, n_layers=2)
    
    print(f"  Environment: {env.size}×{env.size} grid with {env.n_obstacles} obstacles")
    print(f"  Policy: {policy.count_parameters()} parameters")
    
    # Baseline
    print("\n[2/4] Evaluating random baseline...")
    random_policy = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, n_layers=2)
    baseline_reward, baseline_success, baseline_steps = evaluate_policy(
        random_policy, env, n_episodes=20, max_steps=50, deterministic=False
    )
    print(f"  Random policy: reward={baseline_reward:.3f}, success={baseline_success:.1%}, steps={baseline_steps:.1f}")
    
    # Training
    print("\n[3/4] Training with ES (20 iterations, ~2 minutes)...")
    history = train_es(
        policy=policy,
        env=env,
        N=20,
        sigma=0.05,
        alpha=0.01,
        n_iterations=20,
        n_eval_episodes=5,
        max_steps=50,
        eval_every=5,
        verbose=True
    )
    
    # Final evaluation
    print("\n[4/4] Final evaluation...")
    final_reward, final_success, final_steps = evaluate_policy(
        policy, env, n_episodes=50, max_steps=50, deterministic=True
    )
    print(f"  Trained policy: reward={final_reward:.3f}, success={final_success:.1%}, steps={final_steps:.1f}")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Baseline (random): {baseline_success:.1%} success rate")
    print(f"Trained (ES):      {final_success:.1%} success rate")
    print(f"Improvement:       {(final_success - baseline_success)*100:.1f} percentage points")
    print("="*80)
    
    # Save plot
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "quick_demo_training.png"
    
    plot_training_curves(history, save_path=plot_path)
    print(f"\nTraining curves saved to: {plot_path}")
    
    # Visualize a few episodes
    print("\nVisualizing trained policy behavior...")
    for ep in range(3):
        state = env.reset()
        done = False
        steps = 0
        trajectory = [env.agent_pos]
        
        while not done and steps < 50:
            action, _ = policy.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            trajectory.append(env.agent_pos)
            steps += 1
        
        status = "SUCCESS" if info['success'] else "FAILED"
        print(f"  Episode {ep+1}: {status} in {steps} steps")
    
    print("\n✅ Demo complete! Check ./results/ for plots.")
    print("   For full comparison, run: cd tiny-grpo-es && python compare_methods.py")

if __name__ == "__main__":
    main()
