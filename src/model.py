"""
Core optimization code for ES-based RL in GridWorld.

This module contains:
- GridWorld environment (simple and harder variants)
- Policy and Value networks
- Evolution Strategies optimizer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ============================================================================
# ENVIRONMENT
# ============================================================================

class GridWorld:
    """
    Simple GridWorld environment with sparse rewards.
    
    State: (x, y) position on grid (one-hot encoded)
    Actions: 0=up, 1=down, 2=left, 3=right
    Goal: Reach the goal position while avoiding obstacles
    Reward: +1 at goal, -0.1 at obstacles, 0 elsewhere (sparse!)
    """
    
    def __init__(
        self,
        size: int = 8,
        n_obstacles: int = 8,
        max_steps: int = 50,
        seed: Optional[int] = None
    ):
        self.size = size
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        
        # Action space
        self.n_actions = 4
        self.action_to_delta = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        
        # State space
        self.n_states = size * size
        
        # Initialize environment
        self._setup_world()
        self.reset()
        
    def _setup_world(self):
        """Set up obstacles and goal."""
        # Place goal in top-right corner
        self.goal_pos = (0, self.size - 1)
        self.start_pos = (self.size - 1, 0)  # bottom-left
        
        # Place obstacles randomly (avoiding start and goal)
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            x, y = self.rng.randint(0, self.size, size=2)
            pos = (x, y)
            # Don't place obstacles at start or goal
            if pos != self.start_pos and pos != self.goal_pos:
                self.obstacles.add(pos)
    
    def reset(self) -> np.ndarray:
        """Reset to start position (bottom-left)."""
        self.agent_pos = self.start_pos
        self.pos = self.agent_pos  # Alias for backward compatibility
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get state representation as one-hot encoded position."""
        state = np.zeros(self.n_states, dtype=np.float32)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action and return (next_state, reward, done, info)."""
        self.steps += 1
        
        # Compute new position
        dx, dy = self.action_to_delta[action]
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.size - 1)
        new_pos = (new_x, new_y)
        
        # Check for obstacles
        if new_pos in self.obstacles:
            # Hit obstacle: small penalty, stay in place
            reward = -0.1
            # Stay at current position
        else:
            # Valid move
            self.agent_pos = new_pos
            self.pos = self.agent_pos  # Update alias
            reward = 0.0
            
            # Check if reached goal (sparse reward!)
            if self.agent_pos == self.goal_pos:
                reward = 1.0
        
        # Check if done
        done = (self.agent_pos == self.goal_pos) or (self.steps >= self.max_steps)
        
        info = {
            'steps': self.steps,
            'pos': self.agent_pos,
            'success': self.agent_pos == self.goal_pos
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, policy_probs: Optional[np.ndarray] = None):
        """Visualize the gridworld."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.size + 1):
            ax.axhline(i, color='black', linewidth=0.5)
            ax.axvline(i, color='black', linewidth=0.5)
        
        # Draw obstacles
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='red', alpha=0.5))
        
        # Draw goal
        ax.add_patch(plt.Rectangle(
            (self.goal_pos[1], self.goal_pos[0]), 1, 1, 
            color='green', alpha=0.5
        ))
        
        # Draw agent
        ax.add_patch(plt.Circle(
            (self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5), 0.3, 
            color='blue'
        ))
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'GridWorld (Steps: {self.steps})')
        plt.tight_layout()
        return fig


class HarderGridWorld(GridWorld):
    """
    Harder version with keys and doors for long-horizon credit assignment.
    Agent must collect key before reaching goal.
    """
    
    def __init__(
        self,
        size: int = 10,
        n_obstacles: int = 15,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        self.has_key = False
        super().__init__(size, n_obstacles, max_steps, seed)
    
    def _setup_world(self):
        """Set up world with key and locked goal."""
        super()._setup_world()
        # Place key in middle-left area
        self.key_pos = (self.size // 2, 0)
        # Goal in top-right
        self.goal_pos = (0, self.size - 1)
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.has_key = False
        return super().reset()
    
    def _get_state(self) -> np.ndarray:
        """State includes position + key status."""
        # Position (one-hot) + has_key flag
        state = np.zeros(self.n_states + 1, dtype=np.float32)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        state[-1] = float(self.has_key)
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take step, checking for key pickup and locked goal."""
        # Take normal step
        next_state, reward, done, info = super().step(action)
        
        # Check for key pickup
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            reward = 0.1  # Small reward for key
        
        # Override goal reward: only if have key
        if self.agent_pos == self.goal_pos:
            if self.has_key:
                reward = 1.0  # Success!
                info['success'] = True
            else:
                reward = -0.5  # Penalty for reaching goal without key
                done = False  # Can't finish without key
                info['success'] = False
        
        # Update state with key status
        next_state = self._get_state()
        return next_state, reward, done, info


# ============================================================================
# POLICY AND VALUE NETWORKS
# ============================================================================

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
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Sample action from policy.
        
        Args:
            state: numpy array of shape (state_dim,)
            deterministic: if True, return argmax action
            
        Returns:
            action: int
            log_prob: log probability of action (None if deterministic)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
                return action, None
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                return action, log_prob
    
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
            value: (batch_size,) or scalar
        """
        return self.network(state).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
