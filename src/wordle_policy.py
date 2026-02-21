"""
Policy networks for Wordle environment.

This module provides policies that can play Wordle by:
1. Simple discrete policy: maps state embedding to word choice from vocabulary
2. LLM-based policy: uses a small language model to generate guesses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

# Handle both relative and absolute imports
try:
    from .wordle_env import WordVocabulary, WordleState
except ImportError:
    from wordle_env import WordVocabulary, WordleState


class WordleDiscretePolicy(nn.Module):
    """
    Discrete policy for Wordle that selects from a fixed vocabulary.
    
    This is a simpler approach than using an LLM - it maps the state
    embedding to a probability distribution over a fixed set of common
    5-letter words.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3
    ):
        super().__init__()
        
        # Load vocabulary
        self.vocab = WordVocabulary()
        self.action_dim = len(self.vocab)
        self.state_dim = state_dim
        
        # Build network: state embedding -> action logits
        layers = []
        prev_dim = state_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: one logit per word in vocabulary
        layers.append(nn.Linear(prev_dim, self.action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state embedding -> action logits.
        
        Args:
            state_embedding: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            logits: (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state_embedding)
    
    def get_action(
        self,
        state_embedding: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Sample action (word index) from policy.
        
        Args:
            state_embedding: numpy array of shape (state_dim,)
            deterministic: if True, return argmax action
            
        Returns:
            action_idx: integer index of the word to guess
            log_prob: log probability of action (None if deterministic)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_embedding)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action_idx = torch.argmax(probs, dim=-1).item()
                return action_idx, None
            else:
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_idx))
                return action_idx, log_prob
    
    def get_action_word(
        self,
        state_embedding: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Get action as a word (for Wordle gameplay).
        
        Returns:
            word: The guessed word
            log_prob: Log probability of the action
        """
        action_idx, log_prob = self.get_action(state_embedding, deterministic)
        word = self.vocab.action_to_word(action_idx)
        return word, log_prob
    
    def format_action_xml(
        self,
        state: WordleState,
        state_embedding: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate XML-formatted action for Prime Intellect environment.
        
        Returns:
            xml_action: Formatted as <think>...</think><guess>WORD</guess>
            log_prob: Log probability of the action
        """
        word, log_prob = self.get_action_word(state_embedding, deterministic)
        
        # Generate simple reasoning
        think_text = self._generate_think_text(state, word)
        
        xml_action = f"<think>{think_text}</think>\n<guess>{word}</guess>"
        
        return xml_action, log_prob
    
    def _generate_think_text(self, state: WordleState, word: str) -> str:
        """Generate simple reasoning text for the think tag."""
        turn = state.turn_number + 1
        
        if turn == 1:
            return f"Starting with common word {word} to gather information."
        else:
            return f"Based on previous feedback, trying {word}."
    
    def get_action_batch(
        self,
        state_embeddings: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probs and entropy for batch of state-action pairs.
        
        Used in PPO training.
        
        Args:
            state_embeddings: (batch_size, state_dim)
            actions: (batch_size,) - word indices
            
        Returns:
            log_probs: (batch_size,)
            entropy: scalar
        """
        logits = self.forward(state_embeddings)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WordleValueNetwork(nn.Module):
    """
    Value function for Wordle states.
    
    Estimates the expected future reward from a given state.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3
    ):
        super().__init__()
        
        # Build network
        layers = []
        prev_dim = state_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer (scalar value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state_embedding: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            value: (batch_size,) or scalar
        """
        return self.network(state_embedding).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
