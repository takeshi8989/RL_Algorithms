import numpy as np
import random


class OffPolicyNStepSarsa:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, n=3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, env):
        # Random policy: b(a|s) = 1/possible_actions
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        return random.choice(valid_actions)

    def update_q(self, rewards, states, actions, tau, T):
        rho = np.prod([
            1 / (1 / self.n_actions)
            for _ in range(tau + 1, min(tau + self.n, T))
        ])

        G = sum(self.gamma**(i - tau - 1) *
                rewards[i] for i in range(tau + 1, min(tau + self.n, T)))
        if tau + self.n < T:
            G += self.gamma**self.n * \
                self.q_table[states[tau + self.n], actions[tau + self.n]]

        self.q_table[states[tau], actions[tau]] += self.alpha * rho * \
            (G - self.q_table[states[tau], actions[tau]])

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(env)

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
                        next_action = self.choose_action(env)
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
