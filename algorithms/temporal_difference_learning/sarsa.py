import numpy as np
import random


class Sarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(state)

            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(next_state)

                self.update_q(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

    def get_policy(self):
        policy = np.argmax(self.q_table, axis=1)
        return policy

    def get_q_table(self):
        return self.q_table


class ExpectedSarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def expected_value(self, next_state):
        q_values = self.q_table[next_state]
        max_action = np.argmax(q_values)

        expected_q = 0.0
        for action in range(self.n_actions):
            if action == max_action:
                prob = 1 - self.epsilon + (self.epsilon / self.n_actions)
            else:
                prob = self.epsilon / self.n_actions
            expected_q += prob * q_values[action]

        return expected_q

    def update_q(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.expected_value(next_state)
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()

            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                self.update_q(state, action, reward, next_state)

                state = next_state

    def get_policy(self):
        policy = np.argmax(self.q_table, axis=1)
        return policy

    def get_q_table(self):
        return self.q_table
