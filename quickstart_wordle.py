#!/usr/bin/env python
"""
Quick start script for Wordle environment.

This script demonstrates the basic usage of the Wordle environment
and policy without running full training.

Usage:
    python quickstart_wordle.py
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np

from wordle_env import WordleEnvironmentWrapper, load_wordle_environment
from wordle_policy import WordleDiscretePolicy, WordleValueNetwork


def main():
    print("=" * 70)
    print("Wordle Environment Quick Start")
    print("=" * 70)
    
    # 1. Initialize environment
    print("\n[1/5] Initializing environment...")
    env = load_wordle_environment(
        num_train_examples=10,
        num_eval_examples=5,
        use_prime_intellect=False  # Mock mode
    )
    print("✓ Environment created")
    
    # 2. Create policy
    print("\n[2/5] Creating policy network...")
    device = torch.device('cpu')
    policy = WordleDiscretePolicy(
        state_dim=64,
        hidden_dim=128,
        n_layers=3
    ).to(device)
    print(f"✓ Policy created with {policy.count_parameters():,} parameters")
    print(f"✓ Vocabulary size: {len(policy.vocab)} words")
    print(f"  Sample words: {', '.join(policy.vocab.words[:10])}")
    
    # 3. Create value network
    print("\n[3/5] Creating value network...")
    value_net = WordleValueNetwork(
        state_dim=64,
        hidden_dim=128,
        n_layers=3
    ).to(device)
    print(f"✓ Value network created with {value_net.count_parameters():,} parameters")
    
    # 4. Test environment interaction
    print("\n[4/5] Testing environment interaction...")
    state = env.reset()
    print(f"✓ Initial state: turn {state.turn_number}, complete={state.game_complete}")
    
    # Get state embedding
    state_embedding = env.get_state_embedding(state)
    print(f"✓ State embedding shape: {state_embedding.shape}")
    
    # Get policy action
    xml_action, log_prob = policy.format_action_xml(state, state_embedding)
    print(f"✓ Policy generated action")
    print(f"  Action: {xml_action[:80]}...")
    
    # Take step
    next_state, reward, done, info = env.step(xml_action)
    print(f"✓ Environment step completed")
    print(f"  Reward: {reward:.3f}")
    print(f"  Done: {done}")
    print(f"  Info keys: {list(info.keys())}")
    
    # 5. Play a full episode
    print("\n[5/5] Playing a complete episode...")
    state = env.reset()
    episode_reward = 0
    turns = 0
    max_turns = 6
    
    guesses = []
    
    policy.eval()
    with torch.no_grad():
        while not done and turns < max_turns:
            # Get embedding and action
            state_embedding = env.get_state_embedding(state)
            xml_action, _ = policy.format_action_xml(state, state_embedding, deterministic=True)
            
            # Extract word for display
            import re
            match = re.search(r'<guess>(.*?)</guess>', xml_action)
            word = match.group(1) if match else "???"
            guesses.append(word)
            
            # Step
            state, reward, done, info = env.step(xml_action)
            episode_reward += reward
            turns += 1
            
            print(f"  Turn {turns}: {word} -> reward={reward:.3f}")
    
    print(f"\n✓ Episode complete!")
    print(f"  Total turns: {turns}")
    print(f"  Total reward: {episode_reward:.3f}")
    print(f"  Success: {info.get('correct_answer', 0.0) > 0.5}")
    print(f"  All guesses: {' -> '.join(guesses)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Run full training: python src/train_wordle.py --iterations 100")
    print("  2. Open notebook: jupyter notebook notebooks/wordle_training.ipynb")
    print("  3. Run tests: pytest tests/test_wordle.py -v")
    print("\nTo use Prime Intellect backend:")
    print("  - Install: pip install verifiers>=0.1.9")
    print("  - Set use_prime_intellect=True in your code")
    print("=" * 70)


if __name__ == '__main__':
    main()
