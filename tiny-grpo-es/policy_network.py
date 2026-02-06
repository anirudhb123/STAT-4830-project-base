"""
Simple policy network for gridworld environments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Simple feedforward policy network.
    Maps state -> action probabilities
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network
        layers = []
        prev_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            logits: (batch_size, action_dim) or (action_dim,)
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: numpy array of shape (state_dim,)
            deterministic: if True, return argmax action
            
        Returns:
            action: int
            log_prob: log probability of action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                return action, log_prob
            
            return action, None
    
    def get_action_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probs and entropy for batch of state-action pairs.
        Used in PPO training.
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            
        Returns:
            log_probs: (batch_size,)
            entropy: scalar
        """
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, entropy
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ValueNetwork(nn.Module):
    """
    Value function approximator for PPO.
    Maps state -> value estimate
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2
    ):
        super().__init__()
        
        # Build network
        layers = []
        prev_dim = state_dim
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
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
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim) or (state_dim,)
            
        Returns:
            value: (batch_size, 1) or (1,)
        """
        return self.network(state).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_policy():
    """Test policy network."""
    state_dim = 64  # 8x8 grid
    action_dim = 4
    
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=32)
    print(f"Policy network: {policy.count_parameters()} parameters")
    
    # Test forward pass
    state = torch.randn(1, state_dim)
    logits = policy(state)
    print(f"Logits shape: {logits.shape}")
    
    probs = policy.get_action_probs(state)
    print(f"Action probs: {probs}")
    print(f"Sum: {probs.sum()}")
    
    # Test action sampling
    state_np = np.random.randn(state_dim)
    action, log_prob = policy.get_action(state_np)
    print(f"Sampled action: {action}, log_prob: {log_prob}")
    
    # Test value network
    value_net = ValueNetwork(state_dim, hidden_dim=32)
    print(f"\nValue network: {value_net.count_parameters()} parameters")
    value = value_net(state)
    print(f"Value: {value}")


if __name__ == "__main__":
    test_policy()
