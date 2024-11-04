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

    def choose_action(self, env, state):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            return max(valid_actions, key=lambda action: q_values[action])

    def update_q(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(env, state)

            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(env, next_state)

                if next_action is None:
                    break

                self.update_q(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

    def get_policy(self):
        policy = np.argmax(self.q_table, axis=1)
        return policy

    def get_q_table(self):
        return self.q_table

    def get_q_values(self, state):
        return self.q_table[state]


class ExpectedSarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, env, state):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[state]
            return max(valid_actions, key=lambda action: q_values[action])

    def expected_value(self, env, next_state):
        valid_actions = env.get_valid_actions()
        q_values = self.q_table[next_state]
        max_action = max(valid_actions, key=lambda action: q_values[action])

        for action in valid_actions:
            if action == max_action:
                prob = 1 - self.epsilon + (self.epsilon / len(valid_actions))
            else:
                prob = self.epsilon / len(valid_actions)
            expected_q = prob * q_values[action]

        return expected_q

    def update_q(self, env, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.expected_value(env, next_state)
        self.q_table[state, action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()

            done = False
            while not done:
                action = self.choose_action(env, state)
                next_state, reward, done, _ = env.step(action)

                self.update_q(env, state, action, reward, next_state)

                state = next_state

    def get_policy(self):
        policy = np.argmax(self.q_table, axis=1)
        return policy

    def get_q_table(self):
        return self.q_table
