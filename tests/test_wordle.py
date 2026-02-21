"""
Tests for Wordle environment integration.

Run with: pytest tests/test_wordle.py -v
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from wordle_env import (
    WordleEnvironmentWrapper,
    WordleState,
    WordVocabulary,
    load_wordle_environment
)
from wordle_policy import WordleDiscretePolicy, WordleValueNetwork


class TestWordVocabulary:
    """Test word vocabulary."""
    
    def test_vocabulary_creation(self):
        vocab = WordVocabulary()
        assert len(vocab) > 0, "Vocabulary should not be empty"
        assert all(len(word) == 5 for word in vocab.words), "All words should be 5 letters"
    
    def test_word_to_action_conversion(self):
        vocab = WordVocabulary()
        # Test roundtrip conversion
        word = vocab.words[0]
        idx = vocab.word_to_action(word)
        word_back = vocab.action_to_word(idx)
        assert word == word_back, "Roundtrip conversion should preserve word"
    
    def test_vocabulary_contains_crane(self):
        vocab = WordVocabulary()
        assert "CRANE" in vocab.words, "Vocabulary should contain CRANE"


class TestWordleEnvironment:
    """Test Wordle environment wrapper."""
    
    def test_environment_creation(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        assert env is not None
        assert env.max_turns == 6
    
    def test_reset(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        state = env.reset()
        
        assert isinstance(state, WordleState)
        assert state.turn_number == 0
        assert not state.game_complete
        assert len(state.previous_guesses) == 0
    
    def test_step(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        state = env.reset()
        
        # Take a step
        action = "<think>Testing</think><guess>CRANE</guess>"
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(next_state, WordleState)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert next_state.turn_number == 1
    
    def test_state_embedding(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        state = env.reset()
        
        embedding = env.get_state_embedding(state)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32
    
    def test_parse_guess(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        
        # Test XML parsing
        action1 = "<think>reason</think><guess>CRANE</guess>"
        guess1 = env._parse_guess(action1)
        assert guess1 == "CRANE"
        
        # Test without XML
        action2 = "SLATE"
        guess2 = env._parse_guess(action2)
        assert guess2 == "SLATE"
    
    def test_complete_episode(self):
        env = WordleEnvironmentWrapper(num_episodes=10, max_turns=6)
        state = env.reset()
        
        done = False
        turns = 0
        max_turns = 6
        
        while not done and turns < max_turns:
            action = f"<guess>CRANE</guess>"
            state, reward, done, info = env.step(action)
            turns += 1
        
        assert turns <= max_turns
        assert state.turn_number == turns


class TestWordlePolicy:
    """Test Wordle policy network."""
    
    def test_policy_creation(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=3)
        assert policy is not None
        assert policy.state_dim == 64
        assert policy.action_dim == len(policy.vocab)
    
    def test_forward_pass(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=2)
        
        # Test single state
        state = torch.randn(64)
        logits = policy(state)
        assert logits.shape == (len(policy.vocab),)
        
        # Test batch
        batch_states = torch.randn(10, 64)
        batch_logits = policy(batch_states)
        assert batch_logits.shape == (10, len(policy.vocab))
    
    def test_get_action(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=2)
        state = np.random.randn(64).astype(np.float32)
        
        # Test stochastic action
        action_idx, log_prob = policy.get_action(state, deterministic=False)
        assert isinstance(action_idx, int)
        assert 0 <= action_idx < len(policy.vocab)
        assert log_prob is not None
        
        # Test deterministic action
        action_idx_det, log_prob_det = policy.get_action(state, deterministic=True)
        assert isinstance(action_idx_det, int)
        assert log_prob_det is None
    
    def test_get_action_word(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=2)
        state = np.random.randn(64).astype(np.float32)
        
        word, log_prob = policy.get_action_word(state, deterministic=False)
        assert isinstance(word, str)
        assert len(word) == 5
        assert word.isupper()
    
    def test_format_action_xml(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=2)
        state = np.random.randn(64).astype(np.float32)
        wordle_state = WordleState(
            conversation_history="Test",
            turn_number=0,
            game_complete=False
        )
        
        xml_action, log_prob = policy.format_action_xml(
            wordle_state, state, deterministic=False
        )
        
        assert "<think>" in xml_action
        assert "</think>" in xml_action
        assert "<guess>" in xml_action
        assert "</guess>" in xml_action
    
    def test_get_action_batch(self):
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=128, n_layers=2)
        
        batch_size = 10
        states = torch.randn(batch_size, 64)
        actions = torch.randint(0, len(policy.vocab), (batch_size,))
        
        log_probs, entropy = policy.get_action_batch(states, actions)
        
        assert log_probs.shape == (batch_size,)
        assert isinstance(entropy.item(), float)


class TestWordleValue:
    """Test Wordle value network."""
    
    def test_value_creation(self):
        value_net = WordleValueNetwork(state_dim=64, hidden_dim=128, n_layers=3)
        assert value_net is not None
    
    def test_forward_pass(self):
        value_net = WordleValueNetwork(state_dim=64, hidden_dim=128, n_layers=2)
        
        # Test single state
        state = torch.randn(64)
        value = value_net(state)
        assert value.shape == ()
        
        # Test batch
        batch_states = torch.randn(10, 64)
        batch_values = value_net(batch_states)
        assert batch_values.shape == (10,)


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_policy_env_integration(self):
        """Test that policy can interact with environment."""
        env = WordleEnvironmentWrapper(num_episodes=5, max_turns=6)
        policy = WordleDiscretePolicy(state_dim=64, hidden_dim=64, n_layers=2)
        
        # Play one episode
        state = env.reset()
        done = False
        turns = 0
        
        while not done and turns < 6:
            state_embedding = env.get_state_embedding(state)
            action_xml, _ = policy.format_action_xml(
                state, state_embedding, deterministic=True
            )
            state, reward, done, info = env.step(action_xml)
            turns += 1
        
        assert turns <= 6
        assert isinstance(info.get('correct_answer'), float)
    
    def test_load_environment_function(self):
        """Test the load_environment helper function."""
        env = load_wordle_environment(
            num_train_examples=10,
            num_eval_examples=5,
            use_prime_intellect=False
        )
        
        assert env is not None
        assert isinstance(env, WordleEnvironmentWrapper)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
