import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99,
                 lr=1e-3, batch_size=64, memory_size=10000, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Networks
        self.q_network = DQN(state_dim, action_dim, hidden_dim)
        self.target_network = DQN(state_dim, action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(
                next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, targets)

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                # Store the transition
                self.store_transition(state, action, reward, next_state, done)

                # Perform a training step
                self.train_step()

                state = next_state
                total_reward += reward

            # Update the target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_network()

            # Decay epsilon after each episode
            self.decay_epsilon()

            print(f'Episode: {episode + 1}, Total Reward: {total_reward}')
