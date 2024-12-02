import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class TD3:
    def __init__(self, state_dim, action_dim, action_bounds, actor_model, critic_model, actor_lr, critic_lr, gamma, tau, policy_noise, noise_clip, policy_delay, buffer_size, batch_size, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low, self.action_high = float(action_bounds[0]), float(action_bounds[1])
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.max_action = max_action

        self.actor = actor_model(state_dim, action_dim, max_action)
        self.actor_target = actor_model(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = critic_model(state_dim, action_dim)
        self.critic2 = critic_model(state_dim, action_dim)
        self.critic1_target = critic_model(state_dim, action_dim)
        self.critic2_target = critic_model(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.total_updates = 0

    def select_action(self, state, noise_scale=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action, self.action_low, self.action_high)

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        noise = torch.clamp(
            torch.normal(0, self.policy_noise, size=(self.batch_size, self.action_dim)),
            -self.noise_clip, self.noise_clip
        )
        next_actions = self.actor_target(next_states) + noise
        next_actions = torch.clamp(next_actions, self.action_low, self.action_high)

        target_q1, _ = self.critic1_target(next_states, next_actions)
        target_q2, _ = self.critic2_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, _ = self.critic1(states, actions)
        current_q2, _ = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic1.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_updates += 1
