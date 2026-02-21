"""
PPO (Proximal Policy Optimization) training utilities for Wordle.

Adapted to work with Prime Intellect's Wordle environment.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Any, Dict, Optional


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        """Convert to tensors."""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones),
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Returns:
        advantages: (T,)
        returns: (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values
    return advantages, returns


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 5,
    max_steps: int = 6
) -> Tuple[float, float, float]:
    """
    Evaluate policy on Wordle environment.
    
    Args:
        policy: WordleDiscretePolicy or similar
        env: WordleEnvironmentWrapper
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum turns per episode (default 6 for Wordle)
        
    Returns:
        mean_reward: Average episode reward
        mean_success: Average success rate
        mean_steps: Average steps to completion
    """
    total_rewards = []
    successes = []
    episode_steps = []
    
    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                # Convert state to embedding for policy
                state_embedding = env.get_state_embedding(state)
                
                # Get action (word guess)
                if hasattr(policy, 'format_action_xml'):
                    # WordleDiscretePolicy with XML formatting
                    action_xml, _ = policy.format_action_xml(
                        state, state_embedding, deterministic=True
                    )
                    state, reward, done, info = env.step(action_xml)
                else:
                    # Fallback to simple action
                    action_idx, _ = policy.get_action(state_embedding, deterministic=True)
                    word = policy.vocab.action_to_word(action_idx)
                    action_xml = f"<guess>{word}</guess>"
                    state, reward, done, info = env.step(action_xml)
                
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            successes.append(float(info.get('correct_answer', 0.0)))
            episode_steps.append(steps)
    
    policy.train()
    return np.mean(total_rewards), np.mean(successes), np.mean(episode_steps)


def train_ppo_wordle(
    policy,
    value_net,
    env,
    n_iterations: int = 200,
    n_episodes_per_iter: int = 10,
    n_epochs: int = 4,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    lr_policy: float = 3e-4,
    lr_value: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_every: int = 5,
    log_wandb: bool = False,
    seed: int = 42
):
    """
    Train policy using PPO on Wordle environment.
    
    Args:
        policy: WordleDiscretePolicy
        value_net: WordleValueNetwork
        env: WordleEnvironmentWrapper
        n_iterations: Number of training iterations
        n_episodes_per_iter: Episodes to collect per iteration
        n_epochs: Epochs per iteration
        batch_size: Minibatch size
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clip parameter
        lr_policy: Policy learning rate
        lr_value: Value learning rate
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Max gradient norm for clipping
        eval_every: Evaluate every N iterations
        log_wandb: Log to wandb
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = next(policy.parameters()).device
    
    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Tracking
    best_reward = -float('inf')
    ema_reward = None
    total_games = 0
    
    # Training history
    history = {
        'rewards': [],
        'success_rates': [],
        'avg_turns': []
    }
    
    for iteration in range(n_iterations):
        # Collect rollout
        buffer = RolloutBuffer()
        episode_rewards = []
        episode_successes = []
        
        policy.eval()
        value_net.eval()
        
        for episode_idx in range(n_episodes_per_iter):
            state = env.reset()
            episode_reward = 0
            done = False
            turns = 0
            max_turns = 6  # Standard Wordle limit
            
            while not done and turns < max_turns:
                # Convert state to embedding
                state_embedding = env.get_state_embedding(state)
                
                # Get action and value
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_embedding).to(device)
                    
                    # Get action (word guess) with XML formatting
                    if hasattr(policy, 'format_action_xml'):
                        action_xml, log_prob = policy.format_action_xml(
                            state, state_embedding, deterministic=False
                        )
                        # Also get the action index for storage
                        action_idx, _ = policy.get_action(state_embedding, deterministic=False)
                    else:
                        action_idx, log_prob = policy.get_action(state_embedding, deterministic=False)
                        word = policy.vocab.action_to_word(action_idx)
                        action_xml = f"<guess>{word}</guess>"
                    
                    # Get value estimate
                    value = value_net(state_tensor).item()
                
                # Take step in environment
                next_state, reward, done, info = env.step(action_xml)
                
                # Store transition (using embedding as state)
                buffer.add(
                    state_embedding,
                    action_idx,
                    reward,
                    log_prob.item() if log_prob is not None else 0.0,
                    value,
                    done
                )
                
                episode_reward += reward
                turns += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_successes.append(float(info.get('correct_answer', 0.0)))
            total_games += 1
        
        # Skip update if we don't have enough data
        if len(buffer) == 0:
            print(f"Iter {iteration+1}/{n_iterations}: No data collected, skipping update")
            continue
        
        # Convert buffer to tensors
        states, actions, rewards, old_log_probs, values, dones = buffer.get()
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update networks
        policy.train()
        value_net.train()
        
        for epoch in range(n_epochs):
            # Create minibatches
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(buffer), batch_size):
                end = min(start + batch_size, len(buffer))
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute policy loss
                log_probs, entropy = policy.get_action_batch(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy_coef * entropy
                
                # Total policy loss
                total_policy_loss = policy_loss + entropy_loss
                
                # Update policy
                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                policy_optimizer.step()
                
                # Compute value loss
                pred_values = value_net(batch_states)
                value_loss = F.mse_loss(pred_values, batch_returns)
                
                # Update value network
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                value_optimizer.step()
        
        # Tracking
        mean_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_success = np.mean(episode_successes) if episode_successes else 0.0
        
        if ema_reward is None:
            ema_reward = mean_episode_reward
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * mean_episode_reward
        
        if mean_episode_reward > best_reward:
            best_reward = mean_episode_reward
        
        # Store history
        history['rewards'].append(mean_episode_reward)
        history['success_rates'].append(mean_success)
        
        # Evaluation
        if (iteration + 1) % eval_every == 0 or iteration == 0:
            eval_reward, eval_success, eval_steps = evaluate_policy(
                policy, env, n_episodes=10, max_steps=6
            )
            
            history['avg_turns'].append(eval_steps)
            
            print(f"Iter {iteration+1}/{n_iterations}: "
                  f"train_reward={mean_episode_reward:.3f}, "
                  f"train_success={mean_success:.2f}, "
                  f"eval_reward={eval_reward:.3f}, "
                  f"eval_success={eval_success:.2f}, "
                  f"eval_turns={eval_steps:.1f}, "
                  f"ema={ema_reward:.3f}, best={best_reward:.3f}")
            
            if log_wandb:
                try:
                    import wandb
                    wandb.log({
                        'iteration': iteration + 1,
                        'train/reward': mean_episode_reward,
                        'train/success_rate': mean_success,
                        'eval/reward': eval_reward,
                        'eval/success_rate': eval_success,
                        'eval/avg_turns': eval_steps,
                        'train/ema_reward': ema_reward,
                        'train/best_reward': best_reward,
                        'train/total_games': total_games
                    })
                except ImportError:
                    pass
    
    return policy, value_net, history


# Backwards compatibility alias
def train_ppo(
    policy,
    value_net,
    env_class,
    env_kwargs: dict,
    n_iterations: int = 200,
    n_steps: int = 128,
    n_epochs: int = 4,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    lr_policy: float = 3e-4,
    lr_value: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_every: int = 5,
    log_wandb: bool = True,
    seed: int = 42
):
    """
    Train policy using PPO (legacy interface for GridWorld).
    
    For Wordle training, use train_ppo_wordle instead.
    """
    # Create environment
    env = env_class(**env_kwargs)
    
    # Check if this is a Wordle environment
    if hasattr(env, 'get_state_embedding'):
        return train_ppo_wordle(
            policy=policy,
            value_net=value_net,
            env=env,
            n_iterations=n_iterations,
            n_episodes_per_iter=n_steps // 6,  # Approximate conversion
            n_epochs=n_epochs,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            lr_policy=lr_policy,
            lr_value=lr_value,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            eval_every=eval_every,
            log_wandb=log_wandb,
            seed=seed
        )
    
    # Original GridWorld training code
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = next(policy.parameters()).device
    
    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Tracking
    best_reward = -float('inf')
    ema_reward = None
    total_steps = 0
    
    for iteration in range(n_iterations):
        # Collect rollout
        buffer = RolloutBuffer()
        state = env.reset()
        episode_rewards = []
        episode_reward = 0
        
        policy.eval()
        value_net.eval()
        
        for step in range(n_steps):
            # Get action and value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action, log_prob = policy.get_action(state, deterministic=False)
                value = value_net(state_tensor).item()
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            buffer.add(state, action, reward, log_prob, value, done)
            
            episode_reward += reward
            total_steps += 1
            state = next_state
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state = env.reset()
        
        # Convert buffer to tensors
        states, actions, rewards, old_log_probs, values, dones = buffer.get()
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update networks
        policy.train()
        value_net.train()
        
        for epoch in range(n_epochs):
            # Create minibatches
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute policy loss
                log_probs, entropy = policy.get_action_batch(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy_coef * entropy
                
                # Total policy loss
                total_policy_loss = policy_loss + entropy_loss
                
                # Update policy
                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                policy_optimizer.step()
                
                # Compute value loss
                pred_values = value_net(batch_states)
                value_loss = F.mse_loss(pred_values, batch_returns)
                
                # Update value network
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
                value_optimizer.step()
        
        # Tracking
        mean_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        if ema_reward is None:
            ema_reward = mean_episode_reward
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * mean_episode_reward
        
        if mean_episode_reward > best_reward:
            best_reward = mean_episode_reward
        
        # Evaluation
        if (iteration + 1) % eval_every == 0 or iteration == 0:
            max_eval_steps = getattr(env, 'max_steps', 50)
            eval_reward, eval_success, eval_steps = evaluate_policy(
                policy, env, n_episodes=10, max_steps=max_eval_steps
            )
            
            print(f"Iter {iteration+1}/{n_iterations}: "
                  f"train_reward={mean_episode_reward:.3f}, "
                  f"eval_reward={eval_reward:.3f}, eval_success={eval_success:.2f}, "
                  f"eval_steps={eval_steps:.1f}, ema={ema_reward:.3f}, best={best_reward:.3f}")
    
    return policy, value_net
