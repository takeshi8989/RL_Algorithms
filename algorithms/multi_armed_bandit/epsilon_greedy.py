import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def bandit(self, action):
        # Reward function
        # Action number A mod 3 and A mod 7determines the mean
        # of the reward distribution
        action_group = action % 3 + (action % 7) * 0.1
        return np.random.normal(loc=action_group, scale=1)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

    def run(self, num_iterations):
        for _ in range(num_iterations):
            action = self.choose_action()
            reward = self.bandit(action)
            self.update(action, reward)
