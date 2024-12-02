import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class SAC:
    def __init__(self, state_dim, action_dim, action_bounds, actor_model, critic_model, actor_lr, critic_lr, alpha_lr, gamma, tau, max_action, target_entropy=None, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low, self.action_high = float(action_bounds[0]), float(action_bounds[1])
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy or -action_dim
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.max_action = max_action

        # Networks
        self.actor = actor_model(state_dim, action_dim, max_action)
        self.critic1 = critic_model(state_dim, action_dim)
        self.critic2 = critic_model(state_dim, action_dim)
        self.critic1_target = critic_model(state_dim, action_dim)
        self.critic2_target = critic_model(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=alpha_lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 256

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        return torch.tanh(action).detach().numpy()[0]

    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Update critics
        with torch.no_grad():
            next_actions, log_probs = self.actor.sample(next_states)
            target_q1, _ = self.critic1_target(next_states, next_actions)
            target_q2, _ = self.critic2_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (torch.min(target_q1, target_q2) - self.alpha * log_probs)

        current_q1, _ = self.critic1(states, actions)
        current_q2, _ = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q_values = torch.min(
            self.critic1.Q1(states, new_actions),
            self.critic2.Q1(states, new_actions)
        )
        actor_loss = (self.alpha * log_probs - q_values).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        if self.target_entropy:
            alpha_loss = -(self.alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, num_episodes, max_steps):
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)

                self.store_transition((state, action, reward, next_state, done))
                self.update()

                state = next_state
                episode_reward += reward
                if done:
                    break

            print(f"Episode {episode + 1}: Reward = {episode_reward}")
