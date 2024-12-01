import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, action_low, action_high, gamma=0.99, tau=0.005, buffer_capacity=100000, batch_size=64, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Actor network
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic network
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh(),
        )

    def build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def select_action(self, state, noise_scale=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        noise = noise_scale * np.random.randn(self.action_dim)
        action = np.clip(action + noise, self.action_low, self.action_high)
        return action

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute Q-targets
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_targets = rewards + self.gamma * \
                (1 - dones) * self.critic_target(torch.cat([next_states, next_actions], dim=1)).squeeze()

        # Update critic
        q_values = self.critic(torch.cat([states, actions], dim=1)).squeeze()
        critic_loss = ((q_values - q_targets) ** 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actions_pred], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_network(self.actor_target, self.actor)
        self.update_target_network(self.critic_target, self.critic)

    def update_target_network(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, transition):
        self.replay_buffer.add(transition)

    def train(self, env, num_episodes, max_steps):
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                self.store_transition((state, action, reward, next_state, done))
                self.update()
                state = next_state
                total_reward += reward
                if done:
                    break
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
