
import numpy as np
import random


class ExpectedSarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, env, state):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.get_q_values(state)
            return max(valid_actions, key=lambda action: q_values[action])

    def expected_value(self, env, next_state):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return 0
        q_values = self.get_q_values(next_state)
        max_action = max(valid_actions, key=lambda action: q_values[action])

        for action in valid_actions:
            if action == max_action:
                prob = 1 - self.epsilon + (self.epsilon / len(valid_actions))
            else:
                prob = self.epsilon / len(valid_actions)
            expected_q = prob * q_values[action]

        return expected_q

    def update_q(self, env, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        predict = q_values[action]
        target = reward + self.gamma * self.expected_value(env, next_state)
        q_values[action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()

            done = False
            while not done:
                action = self.choose_action(env, state)
                if action is None:
                    break
                next_state, reward, done, _ = env.step(action)

                self.update_q(env, state, action, reward, next_state)

                state = next_state

    def get_q_values(self, state):
        state_key = tuple(map(tuple, state))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def choose_best_action(self, env, state):
        valid_actions = env.get_valid_actions()
        q_values = self.get_q_values(state)
        return max(valid_actions, key=lambda action: q_values[action])
