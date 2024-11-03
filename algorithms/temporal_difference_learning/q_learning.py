import numpy as np
import random


class QLearningAgent:
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

    def update_q(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
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


class DoubleQLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_1 = np.zeros((n_states, n_actions))
        self.q_table_2 = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            q_values = self.q_table_1[state] + self.q_table_2[state]
            return np.argmax(q_values)

    def update_q(self, state, action, reward, next_state):
        if random.uniform(0, 1) < 0.5:
            best_action = np.argmax(self.q_table_1[next_state])
            target = reward + self.gamma * \
                self.q_table_2[next_state, best_action]
            self.q_table_1[state, action] += self.alpha * \
                (target - self.q_table_1[state, action])
        else:
            best_action = np.argmax(self.q_table_2[next_state])
            target = reward + self.gamma * \
                self.q_table_1[next_state, best_action]
            self.q_table_2[state, action] += self.alpha * \
                (target - self.q_table_2[state, action])

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
        combined_q = self.q_table_1 + self.q_table_2
        policy = np.argmax(combined_q, axis=1)
        return policy

    def get_q_tables(self):
        return self.q_table_1, self.q_table_2
