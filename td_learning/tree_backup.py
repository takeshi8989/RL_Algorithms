import numpy as np
import random


class TreeBackup:
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
        if tau + 1 >= T:
            G = rewards[T]
        else:
            G = rewards[tau + 1] + self.gamma * np.sum([
                self.get_policy_probabilities(states[tau + 1])[a] *
                self.q_table[states[tau + 1], a]
                for a in range(self.n_actions)
            ])

        for k in range(min(tau + self.n, T - 1), tau, -1):
            best_action = np.argmax(self.q_table[states[k]])
            G = rewards[k] + self.gamma * np.sum([
                self.get_policy_probabilities(states[k])[a] *
                self.q_table[states[k], a]
                for a in range(self.n_actions)
                if a != best_action
            ]) + self.get_policy_probabilities(states[k])[best_action] * G

        self.q_table[states[tau], actions[tau]] += self.alpha * \
            (G - self.q_table[states[tau], actions[tau]])

    def get_policy_probabilities(self, state):
        probabilities = np.ones(self.n_actions) * \
            (self.epsilon / self.n_actions)
        best_action = np.argmax(self.q_table[state])
        probabilities[best_action] += (1.0 - self.epsilon)
        return probabilities

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
                self.update_q(rewards, states, actions, tau, T)

                if tau == T - 1:
                    break

                t += 1

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_q_table(self):
        return self.q_table
