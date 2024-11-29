import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class PPO:
    def __init__(self, state_dim, action_dim, clip_epsilon, gamma, actor_lr,
                 critic_lr, epochs, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size

        self.actor = self.build_actor_network()
        self.critic = self.build_critic_network()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr)

    def build_actor_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def build_critic_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones):
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def compute_advantages(self, returns, values):
        advantages = returns - values
        return advantages

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                idx = slice(i, i + self.batch_size)
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                action_probs = self.actor(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)

                ratios = torch.exp(log_probs - batch_old_log_probs)

                surrogate1 = ratios * batch_advantages
                surrogate2 = torch.clamp(
                    ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                values = self.critic(batch_states).squeeze()
                value_loss = (batch_returns - values).pow(2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

    def train(self, env, num_iterations, timesteps_per_batch):
        for iteration in range(num_iterations):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state, _ = env.reset()
            episode_reward = 0
            total_rewards = []
            done = False

            for _ in range(timesteps_per_batch):
                action, log_prob = self.select_action(state)
                value = self.critic(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ).item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.item())
                dones.append(done)
                values.append(value)

                state = next_state if not done else env.reset()[0]
                episode_reward += reward

                if done:
                    total_rewards.append(episode_reward)
                    episode_reward = 0

            states = torch.tensor(np.array(states), dtype=torch.float32)
            returns = self.compute_returns(rewards, dones)
            values = torch.tensor(values, dtype=torch.float32)
            returns = returns.clone().detach()
            advantages = self.compute_advantages(returns, values) \
                .clone().detach()

            self.update(states, actions, log_probs, returns, advantages)

            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"Iteration {iteration + 1}: "
                  f"Average Reward = {avg_reward:.2f}")
