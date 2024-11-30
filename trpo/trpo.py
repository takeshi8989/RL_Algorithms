import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class TRPO:
    def __init__(self, state_dim, action_dim, gamma, delta, alpha, max_backtracking_steps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.max_backtracking_steps = max_backtracking_steps

        self.policy = self.build_policy_network()
        self.old_policy = self.build_policy_network()
        self.policy_optimizer = None

        self.value_function = self.build_value_network()
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=3e-4)

    def build_policy_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def build_value_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, dones):
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def conjugate_gradient(self, Hx, b, n_steps=10, tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        for _ in range(n_steps):
            Hp = Hx(p)
            alpha = torch.dot(r, r) / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            r_new = r - alpha * Hp
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = r_new + beta * p
            if r.norm() < tol:
                break
        return x

    def hessian_vector_product(self, states, actions, vector, damping=1e-2):
        action_probs = self.policy(states)
        dist = Categorical(action_probs)

        with torch.no_grad():
            old_action_probs = self.old_policy(states)
        old_dist = Categorical(old_action_probs)

        kl = torch.distributions.kl_divergence(old_dist, dist).mean()

        grads = torch.autograd.grad(
            outputs=kl,
            inputs=self.policy.parameters(),
            create_graph=True
        )

        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        grad_vector_product = torch.dot(flat_grads, vector)

        hessian_vector_grads = torch.autograd.grad(
            outputs=grad_vector_product,
            inputs=self.policy.parameters(),
            retain_graph=True
        )

        hessian_vector_product = torch.cat([grad.view(-1) for grad in hessian_vector_grads])
        return hessian_vector_product + damping * vector

    def line_search(self, states, actions, advantages, step_dir, max_kl):
        old_policy_params = list(self.policy.parameters())
        step_size = 1.0

        for _ in range(self.max_backtracking_steps):
            with torch.no_grad():
                for param, step in zip(self.policy.parameters(), step_dir):
                    param.data += step_size * step

                kl = self.compute_kl_divergence(states)
                if kl <= max_kl:
                    return True

                step_size *= self.alpha

            for param, old_param in zip(self.policy.parameters(), old_policy_params):
                param.data.copy_(old_param.data)

        return False

    def compute_kl_divergence(self, states):
        with torch.no_grad():
            old_action_probs = self.old_policy(states)
        old_dist = Categorical(old_action_probs)

        action_probs = self.policy(states)
        current_dist = Categorical(action_probs)

        kl_divergence = torch.distributions.kl_divergence(old_dist, current_dist)
        return kl_divergence.mean()

    def train(self, env, num_iterations, timesteps_per_batch):
        for iteration in range(num_iterations):
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            state, _ = env.reset()
            done = False

            for _ in range(timesteps_per_batch):
                action, log_prob = self.select_action(state)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                value = self.value_function(state_tensor).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)

                state = next_state if not done else env.reset()[0]

            returns, advantages = self.compute_advantages(rewards, values, dones)
            log_probs = torch.stack(log_probs)
            states = torch.tensor(np.array(states), dtype=torch.float32)

            g = torch.autograd.grad(
                outputs=(log_probs * advantages).mean(),
                inputs=list(self.policy.parameters()),
                retain_graph=True
            )

            g = torch.cat([grad.view(-1) for grad in g])

            step_dir = self.conjugate_gradient(
                lambda v: self.hessian_vector_product(states, actions, v), g
            )

            self.line_search(states, actions, advantages, step_dir, self.delta)

            values_tensor = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            value_loss = ((returns - values_tensor) ** 2).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            total_reward = sum(rewards)
            kl = self.compute_kl_divergence(states).item()

            print(
                f"Iteration {iteration + 1}: Total Reward = {total_reward}, Value Loss = {value_loss.item()}, KL Divergence = {kl}")
