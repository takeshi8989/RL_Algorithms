import torch
import torch.nn as nn
import torch.optim as optim


class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Policy network
        self.policy = self.build_policy_network()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def build_policy_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action)

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_policy(self, log_probs, returns):
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for numerical stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            # Negative because we want to maximize the reward
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        # Perform gradient descent
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state, _ = env.reset()
            log_probs = []
            rewards = []

            done = False

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            returns = self.compute_returns(rewards)
            self.update_policy(log_probs, returns)

            total_reward = sum(rewards)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
