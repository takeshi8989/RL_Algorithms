import numpy as np
import random


class NStepSarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=0.1, n=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
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

    def update_q(self, rewards, states, actions, tau, T):
        G = sum(self.gamma**(i - tau - 1) *
                rewards[i] for i in range(tau + 1, min(tau + self.n, T)))
        if tau + self.n < T:
            G += self.gamma**self.n * \
                self.q_table[states[tau + self.n], actions[tau + self.n]]

        self.q_table[states[tau], actions[tau]] += self.alpha * \
            (G - self.q_table[states[tau], actions[tau]])

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(env, state)

            rewards = [0]
            states = [state]
            actions = [action]

            T = float('inf')
            t = 0
            while True:
                if t < T:
                    next_state, reward, done, _ = env.step(action)
                    rewards.append(reward)
                    states.append(next_state)

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(env, next_state)
                        actions.append(next_action)
                        action = next_action

                tau = t - self.n + 1
                if tau >= 0:
                    self.update_q(rewards, states, actions, tau, T)

                if tau == T - 1:
                    break

                t += 1

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_q_table(self):
        return self.q_table
