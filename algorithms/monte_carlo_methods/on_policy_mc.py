import numpy as np
from collections import defaultdict


class OnPolicyMC:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = defaultdict(
            lambda: {a: epsilon/len(actions) for a in actions})

    def update_policy(self, state):
        best_action = max(self.actions, key=lambda a: self.Q[state][a])

        for action in self.actions:
            if action == best_action:
                self.policy[state][action] = 1 - self.epsilon + \
                    self.epsilon / len(self.actions)
            else:
                self.policy[state][action] = self.epsilon / len(self.actions)

    def choose_action(self, state):
        probs = list(self.policy[state].values())
        return np.random.choice(self.actions, p=probs)

    def update(self, episode):
        states, actions, rewards = zip(*episode)

        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action = states[t], actions[t]
            G = self.gamma * G + rewards[t]

            if (state, action) not in zip(states[:t], actions[:t]):
                self.returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns[state][action])
                self.update_policy(state)
