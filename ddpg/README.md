# Deep Deterministic Policy Gradient (DDPG) 

**Initialize**:

- Actor network $\pi_{\theta}(a|s)$ with parameters $\theta$
- Critic network $Q_{\phi}(a|s)$ with parameters $\phi$
- Target networks $\pi_{\theta_{\text{target}}}(a|s)$ and $Q_{\phi_{\text{target}}}(a|s)$ initialized with $\theta_{\text{target}} = \theta, \phi_{\text{target}} = \phi$
- Replay Buffer $D$
- Hyperparameters
  - Discount factor $\gamma$
  - Soft update coefficient $\tau$
  - Batch size $B$

**Loop for each iteration**:

- For each timestep:
    - Initialize environment and observe initial state  $s_0$
    - Select action $a_t = \pi_{\theta}(s_t) + \epsilon$, where $\epsilon \sim N(0, \sigma)$ 
    - Execute $a_t$ in the environment and observe reward $r_t$, next state $s_{t+1}$, and done signal $d_t$.
    - Store $(s_t, a_t, r_t, s_{t+1}, d_t)$ in $D$.

- Sample a minibatch of transitions $(s, a, r, s', d)$ from $D$
- Compute target value: $y(r, s', d) = r + \gamma (1-d) Q_{\phi_{\text{target}}}(s', \pi_{\theta_{\text{target}}}(s'))$
- Minimize Bellman error: $\phi \gets \arg \min\limits_\phi \frac {1}{B} \sum\limits_{{s, a, r, s', d} \in D} (Q_{\phi}(s, a) - y(r, s', d))^2$
- Compute policy gradient: $\theta \gets \arg \max\limits_{\theta} \frac {1}{B} \sum\limits_{s \in D} Q_{\phi}(s, \pi(s))$
- Update target critic network: $\phi_{\text{target}} \gets \tau \phi + (1 - \tau) \phi_{\text{target}}$
- Update target actor network: $\theta_{\text{target}} \gets \tau \theta + (1 - \tau) \theta_{\text{target}}$


## Notes

This algorithm is designed to handle problems with continuous action spaces, which cannot be directly addressed by traditional RL algorithms like Q-learning or DQN. In continuous action spaces, the number of possible actions is infinite, making it computationally infeasible to find the optimal action by enumeration.

### Continuous Action Space Problem

- Continuous Policy Output: The The actor outputs actions within the continuous action bounds $[a_{\text{low}}, a_{\text{high}}]$.

$$
a = clip(\pi_{\theta}(s) + \epsilon,a_{\text{low}}, a_{\text{high}})
$$

Where $\epsilon \sim N(0, \sigma)$ adds Gaussian noise for exploration


- The policy $\pi_{\theta}(s)$ is updated to maximize the critic's Q-value:

$$
\theta \gets \arg \max\limits_{\theta} E[Q_{\theta}(s, \pi_{\theta}(s))]
$$

- The critic network learns using the Bellman equation:

$$
Q(s, a) = E[r + \gamma Q_{\phi_{\text{target}}}(s', \pi_{\theta_{\text{target}}}(s))]
$$