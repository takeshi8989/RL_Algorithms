import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class TRPO:
    def __init__(self, state_dim, action_dim, gamma, delta, alpha,
                 max_backtracking_steps):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.delta = delta  # KL-divergence limit
        self.alpha = alpha  # Backtracking coefficient
        self.max_backtracking_steps = max_backtracking_steps

        # Policy network
        self.policy = self.build_policy_network()
        self.old_policy = self.build_policy_network()
        self.policy_optimizer = None  # TRPO doesn't directly use optimizers

        # Value function
        self.value_function = self.build_value_network()
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(), lr=3e-4)

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
        returns, advantages = [], []
        G = 0
        for reward, value, done in zip(
            reversed(rewards), reversed(values), reversed(dones)
        ):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - values
        return returns, advantages

    def conjugate_gradient(self, Hx, b, n_steps=10, tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        for _ in range(n_steps):
            Hp = Hx(p)
            alpha = (r @ r) / (p @ Hp)
            x = x + alpha * p
            r_new = r - alpha * Hp
            if torch.norm(r_new) < tol:
                break
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new
        return x

    def hessian_vector_product(self, states, actions, vector):
        # Compute the Hessian-vector product for KL divergence
        # Placeholder implementation
        return vector  # Replace this with the actual computation

    def line_search(self, states, actions, advantages, step_dir, max_kl):
        old_policy_params = list(self.policy.parameters())
        step_size = 1.0
        for _ in range(self.max_backtracking_steps):
            with torch.no_grad():
                # Apply the update
                for param, step in zip(self.policy.parameters(), step_dir):
                    param.data += step_size * step

                # Check KL-divergence constraint
                kl = self.compute_kl_divergence(states)
                if kl <= max_kl:
                    return True
                step_size *= self.alpha

            # Revert the parameters
            for param, old_param in zip(
                    self.policy.parameters(), old_policy_params):
                param.data.copy_(old_param.data)

        return False

    def compute_kl_divergence(self, states):
        # Compute the KL divergence between the old policy and the new policy
        # Placeholder implementation
        return 0.0  # Replace this with the actual KL computation

    def train(self, env, num_iterations, timesteps_per_batch):
        for iteration in range(num_iterations):
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            state, _ = env.reset()
            done = False

            # Collect trajectories
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

            # Compute advantages
            returns, advantages = self.compute_advantages(
                rewards, values, dones)

            # Compute policy gradient
            g = torch.autograd.grad(
                outputs=(log_probs * advantages).mean(),
                inputs=list(self.policy.parameters()),
                retain_graph=True
            )

            # Compute search direction via conjugate gradient
            step_dir = self.conjugate_gradient(
                lambda v: self.hessian_vector_product(states, actions, v), g
            )

            # Backtracking line search to update policy
            self.line_search(states, actions, advantages, step_dir, self.delta)

            # Update value function
            value_loss = ((returns - torch.tensor(values)) ** 2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            print(f"Iteration {iteration + 1}: "
                  f"Value Loss = {value_loss.item()}")
