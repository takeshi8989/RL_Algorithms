import numpy as np
import random


class DoubleQLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_1 = {}
        self.q_table_2 = {}

    def choose_action(self, env, state):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values_1 = self.get_q_values(state, self.q_table_1)
            q_values_2 = self.get_q_values(state, self.q_table_2)
            q_values = q_values_1 + q_values_2
            return max(valid_actions, key=lambda action: q_values[action])

    def update_q(self, env, state, action, reward, next_state,
                 next_action=None):
        q_values_1 = self.get_q_values(state, self.q_table_1)
        q_values_2 = self.get_q_values(state, self.q_table_2)
        next_q_values_1 = self.get_q_values(next_state, self.q_table_1)
        next_q_values_2 = self.get_q_values(next_state, self.q_table_2)

        if random.uniform(0, 1) < 0.5:
            best_action = np.argmax(next_q_values_1)
            target = reward + self.gamma * next_q_values_2[best_action]
            q_values_1[action] += self.alpha * (target - q_values_1[action])
        else:
            best_action = np.argmax(next_q_values_2)
            target = reward + self.gamma * next_q_values_1[best_action]
            q_values_2[action] += self.alpha * (target - q_values_2[action])

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(env, state)
                next_state, reward, done, _ = env.step(action)

                self.update_q(env, state, action, reward, next_state)

                state = next_state

    def get_q_values(self, state, q_table):
        state_key = tuple(map(tuple, state))
        if state_key not in q_table:
            q_table[state_key] = np.zeros(self.n_actions)
        return q_table[state_key]

    def choose_best_action(self, env, state):
        valid_actions = env.get_valid_actions()
        q_values_1 = self.get_q_values(state, self.q_table_1)
        q_values_2 = self.get_q_values(state, self.q_table_2)
        q_values = q_values_1 + q_values_2
        return max(valid_actions, key=lambda action: q_values[action])
