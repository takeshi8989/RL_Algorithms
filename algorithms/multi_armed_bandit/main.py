import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def bandit(self, action):
        # Reward function
        # Action number A mod 3 determines the mean of the reward distribution
        action_group = action % 3 
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


if __name__ == "__main__":
    k = 10
    epsilon = 0.1
    num_iterations = 1000

    bandit = EpsilonGreedyBandit(k, epsilon)
    bandit.run(num_iterations)

    print("Estimated action values (Q):", bandit.Q)
    print("Number of times each action was taken (N):", bandit.N)
