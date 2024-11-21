import numpy as np


class DP:
    def __init__(self, states, actions, transition_probs, rewards, gamma=0.9):
        self.states = states
        self.actions = actions
        self.P = transition_probs
        self.R = rewards
        self.gamma = gamma
        self.policy = {s: np.random.choice(actions) for s in states}
        self.V = np.zeros(len(states))

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in self.states:
                v = self.V[s]
                # Deterministic policy
                a = self.policy[s]
                self.V[s] = sum(
                    self.P(s, a, s_prime) *
                    (self.R(s, a) + self.gamma * self.V[s_prime])
                    for s_prime in self.states
                )
                delta = max(delta, abs(v - self.V[s]))

            if delta < 1e-6:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in self.states:
            old_action = self.policy[s]
            self.policy[s] = max(
                self.actions,
                key=lambda a: sum(
                    self.P(s, a, s_prime) *
                    (self.R(s, a) + self.gamma * self.V[s_prime])
                    for s_prime in self.states
                )
            )
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def get_optimal_policy(self):
        return self.policy

    def get_optimal_value_function(self):
        return self.V
