import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple

class Memory:
    """Simplified experience replay buffer"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        """Clear the buffer"""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def add(self, state, action, logprob, reward, done):
        """Add experience"""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)

class PolicyNetwork(nn.Module):
    """Simplified policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass"""
        action_probs = self.policy_net(state)
        state_value = self.value_net(state)
        return action_probs, state_value
    
    def get_action(self, state, valid_actions=None):
        """Get action"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state)

        if valid_actions is not None:
            mask = torch.zeros_like(action_probs)
            for idx in valid_actions:
                mask[0, idx] = 1
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum() + 1e-10)

        dist = Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        return action.item(), logprob

class PPO:
    """Simplified PPO algorithm"""
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
    
    def update(self, memory):
        """Update policy"""
        if len(memory.states) == 0:
            return
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.long).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        for _ in range(10):  
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)

            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            advantages = rewards - state_values.squeeze().detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values.squeeze(), rewards)
            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def get_action(self, state, valid_actions=None):
        """Get action using old policy"""
        return self.policy_old.get_action(state, valid_actions)

def train_factor_agent(env, max_episodes=50, update_frequency=500, verbose=True):

    ppo = PPO(env.state_dim, env.action_dim)
    memory = Memory()

    episode_rewards = []
    best_ic = 0.0
    step_count = 0
    
    print(f"Starting training...")
    print(f"State dimension: {env.state_dim}, Action dimension: {env.action_dim}")
    print(f"Target number of factors: {env.num_factors}")
    print("-" * 50)
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while not env.episode_done and episode_steps < 200:  
            step_count += 1
            episode_steps += 1

            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action, logprob = ppo.get_action(state, valid_actions)

            next_state, reward, done, info = env.step(action)

            memory.add(state, action, logprob, reward, done)
            
            state = next_state
            episode_reward += reward

            if step_count % update_frequency == 0:
                ppo.update(memory)
                memory.clear()
        
        episode_rewards.append(episode_reward)

        if env.factor_pool:
            current_best = max([abs(ic) for ic in env.factor_pool_ics])
            if current_best > best_ic:
                best_ic = current_best

        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:3d}: ")
            print(f"  Average reward: {avg_reward:8.4f}")
            print(f"  Factor pool size: {len(env.factor_pool):2d}/{env.num_factors}")
            print(f"  Best |IC|: {best_ic:8.6f}")
            
            if env.factor_pool_ics:
                top_ics = sorted([abs(ic) for ic in env.factor_pool_ics], reverse=True)[:3]
                print(f"  Top3 |IC|: {[f'{ic:.4f}' for ic in top_ics]}")
            print()
    
    print("Training completed!")
    return ppo

def test_trained_agent(env, ppo, num_test_episodes=3):
    """Test trained agent"""
    print("\n" + "="*50)
    print("Testing trained agent")
    print("="*50)
    
    for episode in range(num_test_episodes):
        print(f"\nTest Episode {episode + 1}")
        print("-" * 30)
        
        env.reset()  
        state = env._get_state()
        steps = 0

        while len(env.factor_pool) < env.num_factors * 1.5 and steps < 100:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action, _ = ppo.get_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            if info.get('factor_done', False):
                print(f"  Generated new factor, reward: {reward:.4f}")
            
            state = next_state
            steps += 1
            
            if env.episode_done:
                break
        
        print(f"  Final factor pool size: {len(env.factor_pool)}")
    
    return env.generate_summary()
