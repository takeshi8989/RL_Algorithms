import numpy as np
from collections import defaultdict


class OffPolicyMC:
    def __init__(self, states, actions, gamma=0.9):
        self.states = states
        self.actions = actions
        self.gamma = gamma

        self.Q = defaultdict(lambda: defaultdict(float))
        self.C = defaultdict(lambda: defaultdict(float))
        self.target_policy = defaultdict(lambda: None)

    def target_policy_action(self, state):
        if self.target_policy[state] is None:
            self.target_policy[state] = max(self.actions,
                                            key=lambda a: self.Q[state][a])
        return self.target_policy[state]

    def behavior_policy_action(self, state):
        return np.random.choice(self.actions)

    def update(self, episode):
        if not episode:
            return

        states, actions, rewards = zip(*episode)

        G = 0
        W = 1

        for t in range(len(episode)-1, -1, -1):
            state, action = states[t], actions[t]
            G = self.gamma * G + rewards[t]

            self.C[state][action] += W
            self.Q[state][action] += (W / self.C[state]
                                      [action]) * (G - self.Q[state][action])

            self.target_policy[state] = max(self.actions,
                                            key=lambda a: self.Q[state][a])

            if action != self.target_policy[state]:
                break

            # π(a|s)=1 if a=argmax_a Q(s,a)
            # b(a|s)=1/|A| because we are using a uniform random policy
            W *= 1.0 / len(self.actions)
