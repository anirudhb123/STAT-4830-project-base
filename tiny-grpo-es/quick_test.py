"""
Quick test script to verify installation and run a minimal experiment.
This should complete in ~2-3 minutes.
"""
import numpy as np
import torch
from gridworld_env import GridWorld
from policy_network import PolicyNetwork
from train_es_gridworld import train_es, evaluate_policy


def test_environment():
    """Test that environment works."""
    print("Testing environment...")
    env = GridWorld(size=5, n_obstacles=3, seed=42)
    
    state = env.reset()
    print(f"  ✓ State shape: {state.shape}")
    print(f"  ✓ Start: {env.pos}, Goal: {env.goal_pos}")
    
    # Random episode
    done = False
    steps = 0
    while not done and steps < 50:
        action = np.random.randint(0, 4)
        state, reward, done, info = env.step(action)
        steps += 1
    
    print(f"  ✓ Random episode completed: {steps} steps, success={info['success']}")
    print()


def test_policy_network():
    """Test that policy network works."""
    print("Testing policy network...")
    
    state_dim = 25  # 5x5 grid
    action_dim = 4
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=32)
    
    print(f"  ✓ Policy created: {policy.count_parameters()} parameters")
    
    # Forward pass
    state = np.random.randn(state_dim)
    action, log_prob = policy.get_action(state)
    
    print(f"  ✓ Forward pass works: action={action}")
    print()


def quick_es_test():
    """Run a quick ES training test (10 iterations)."""
    print("Running quick ES test (10 iterations)...")
    print("This will take ~1-2 minutes...\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Small environment for speed
    env_class = GridWorld
    env_kwargs = {"size": 5, "n_obstacles": 3, "max_steps": 30, "seed": 42}
    
    # Create policy
    dummy_env = env_class(**env_kwargs)
    state_dim = dummy_env._get_state().shape[0]
    action_dim = dummy_env.n_actions
    
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=32, n_layers=2).to(device)
    
    # Evaluate before training
    print("Before training:")
    env = env_class(**env_kwargs)
    reward_before, success_before, steps_before = evaluate_policy(
        policy, env, n_episodes=10, max_steps=30
    )
    print(f"  Reward: {reward_before:.3f}, Success: {success_before:.2f}, Steps: {steps_before:.1f}\n")
    
    # Train with ES (just 10 iterations for quick test)
    policy = train_es(
        policy=policy,
        env_class=env_class,
        env_kwargs=env_kwargs,
        N=10,  # Small population for speed
        sigma=0.05,
        alpha=0.01,
        T=10,  # Just 10 iterations
        n_episodes=3,
        max_steps=30,
        eval_every=5,
        log_wandb=False,  # Disable wandb for quick test
        seed=42
    )
    
    # Evaluate after training
    print("\nAfter training:")
    env = env_class(**env_kwargs)
    reward_after, success_after, steps_after = evaluate_policy(
        policy, env, n_episodes=10, max_steps=30
    )
    print(f"  Reward: {reward_after:.3f}, Success: {success_after:.2f}, Steps: {steps_after:.1f}\n")
    
    # Check improvement
    improvement = reward_after - reward_before
    if improvement > 0:
        print(f"✓ SUCCESS! Policy improved by {improvement:.3f} reward")
        print(f"  (Success rate: {success_before:.2f} → {success_after:.2f})")
    else:
        print(f"⚠ Policy did not improve (might need more iterations)")
        print(f"  This is normal for such a short test - try 50+ iterations")
    
    print("\n" + "="*60)
    print("Quick test complete! Everything is working.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run full comparison: python compare_methods.py --trials 3 --iterations 100")
    print("  2. Or train ES longer: python train_es_gridworld.py")
    print("  3. Or train PPO: python train_ppo_gridworld.py")


def main():
    print("="*60)
    print("QUICK TEST: Verify installation and basic functionality")
    print("="*60)
    print()
    
    test_environment()
    test_policy_network()
    quick_es_test()


if __name__ == "__main__":
    main()
