import torch
import torch.nn as nn
import torch.optim as optim


class A2C:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.actor = self.build_actor_network()
        self.critic = self.build_critic_network()

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate)

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
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def compute_advantage(self, reward, next_state, state):
        state_value = self.critic(torch.tensor(
            state, dtype=torch.float32).unsqueeze(0))
        next_state_value = self.critic(torch.tensor(
            next_state, dtype=torch.float32).unsqueeze(0))
        advantage = reward + self.gamma * next_state_value.item() \
            - state_value.item()

        return advantage, state_value

    def update(self, log_prob, advantage, state_value, reward, next_state):
        actor_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        next_value = self.critic(torch.tensor(
            next_state, dtype=torch.float32).unsqueeze(0))
        target_value = reward + self.gamma * next_value
        critic_loss = (target_value - state_value) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, log_prob = self.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                advantage, state_value = self.compute_advantage(
                    reward, next_state, state)

                self.update(log_prob, advantage,
                            state_value, reward, next_state)

                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
