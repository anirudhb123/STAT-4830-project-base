"""
Simple GridWorld environment for testing ES vs action-space RL.
Features:
- Discrete state and action spaces
- Sparse, outcome-only rewards (jagged reward landscape)
- Long-horizon credit assignment
- Obstacles and multiple goal configurations
"""
import numpy as np
from typing import Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class GridWorld:
    """
    Simple GridWorld environment with sparse rewards.
    
    State: (x, y) position on grid
    Actions: 0=up, 1=down, 2=left, 3=right
    Goal: Reach the goal position while avoiding obstacles
    Reward: Only at goal (+1), small penalty at obstacles (-0.1), 0 elsewhere
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
        
        # Place obstacles randomly (avoiding start and goal)
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            x, y = self.rng.randint(0, self.size, size=2)
            pos = (x, y)
            # Don't place obstacles at start (bottom-left) or goal
            if pos != (self.size - 1, 0) and pos != self.goal_pos:
                self.obstacles.add(pos)
    
    def reset(self) -> np.ndarray:
        """Reset to start position (bottom-left)."""
        self.pos = (self.size - 1, 0)
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get state representation as one-hot encoded position."""
        state = np.zeros(self.n_states, dtype=np.float32)
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action and return (next_state, reward, done, info)."""
        self.steps += 1
        
        # Compute new position
        dx, dy = self.action_to_delta[action]
        new_x = np.clip(self.pos[0] + dx, 0, self.size - 1)
        new_y = np.clip(self.pos[1] + dy, 0, self.size - 1)
        new_pos = (new_x, new_y)
        
        # Check for obstacles
        if new_pos in self.obstacles:
            # Hit obstacle: small penalty, stay in place
            reward = -0.1
            # Stay at current position
        else:
            # Valid move
            self.pos = new_pos
            reward = 0.0
            
            # Check if reached goal (sparse reward!)
            if self.pos == self.goal_pos:
                reward = 1.0
        
        # Check if done
        done = (self.pos == self.goal_pos) or (self.steps >= self.max_steps)
        
        info = {
            'steps': self.steps,
            'pos': self.pos,
            'success': self.pos == self.goal_pos
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
            (self.pos[1] + 0.5, self.pos[0] + 0.5), 0.3, 
            color='blue'
        ))
        
        # Draw policy arrows if provided
        if policy_probs is not None:
            arrow_props = dict(arrowstyle='->', lw=1.5, color='purple', alpha=0.6)
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) not in self.obstacles and (x, y) != self.goal_pos:
                        state = np.zeros(self.n_states, dtype=np.float32)
                        state[x * self.size + y] = 1.0
                        # Get action probabilities for this state
                        # This assumes policy_probs is a function
                        try:
                            probs = policy_probs(state)
                            best_action = np.argmax(probs)
                            dx, dy = self.action_to_delta[best_action]
                            ax.annotate('', 
                                       xy=(y + 0.5 + dy*0.3, x + 0.5 + dx*0.3),
                                       xytext=(y + 0.5, x + 0.5),
                                       arrowprops=arrow_props)
                        except:
                            pass
        
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
    
    def __init__(self, size: int = 10, n_obstacles: int = 15, max_steps: int = 100, seed: Optional[int] = None):
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
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        state[-1] = float(self.has_key)
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take step, checking for key pickup and locked goal."""
        # Take normal step
        next_state, reward, done, info = super().step(action)
        
        # Check for key pickup
        if self.pos == self.key_pos and not self.has_key:
            self.has_key = True
            reward = 0.1  # Small reward for key
        
        # Override goal reward: only if have key
        if self.pos == self.goal_pos:
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
    
    def render(self, policy_probs: Optional[np.ndarray] = None):
        """Render with key visualization."""
        fig = super().render(policy_probs)
        ax = fig.axes[0]
        
        # Draw key
        if not self.has_key:
            ax.add_patch(plt.Rectangle(
                (self.key_pos[1], self.key_pos[0]), 1, 1,
                color='yellow', alpha=0.5
            ))
            ax.text(self.key_pos[1] + 0.5, self.key_pos[0] + 0.5, '🔑',
                   ha='center', va='center', fontsize=20)
        
        # Draw lock on goal if no key
        if not self.has_key:
            ax.text(self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5, '🔒',
                   ha='center', va='center', fontsize=20)
        
        return fig


def test_env():
    """Test the environment."""
    env = GridWorld(size=5, n_obstacles=3, seed=42)
    print("Simple GridWorld:")
    print(f"State shape: {env._get_state().shape}")
    print(f"Actions: {env.n_actions}")
    print(f"Start: {env.pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")
    
    # Run random episode
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = np.random.randint(0, env.n_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Random policy: {steps} steps, reward={total_reward:.2f}, success={info['success']}")
    
    # Test harder environment
    print("\nHarder GridWorld (with key):")
    env2 = HarderGridWorld(size=6, n_obstacles=5, seed=42)
    print(f"State shape: {env2._get_state().shape}")
    print(f"Key position: {env2.key_pos}")


if __name__ == "__main__":
    test_env()
