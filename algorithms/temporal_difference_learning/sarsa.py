import numpy as np
import random


class Sarsa:
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

    def update_q(self, env, state, action, reward, next_state, next_action):
        q_values = self.get_q_values(state)
        predict = q_values[action]
        next_q_values = self.get_q_values(next_state)
        if next_action is None:
            target = reward
        else:
            target = reward + self.gamma * next_q_values[next_action]
        q_values[action] += self.alpha * (target - predict)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(env, state)

            done = False
            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.choose_action(env, next_state)

                self.update_q(env, state, action, reward,
                              next_state, next_action)

                state = next_state
                action = next_action

    def get_q_values(self, state):
        state_key = tuple(map(tuple, state))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]
    
    def get_best_action(self, state):
        q_values = self.get_q_values(state)
        return max(range(self.n_actions), key=lambda action: q_values[action])
