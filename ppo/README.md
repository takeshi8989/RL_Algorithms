# Proximal Policy Optimization (PPO)

**Initialize**:

- Policy network $\pi_{\theta}(a|s)$ with parameters $\theta$
- Value function $V_{\phi}(s)$ with parameters $\phi$
- Hyperparameters
  - Clipping parameter $\epsilon$
  - Learning rates $a_{\theta}$ and $a_{\phi}$
  - Discount factor $\gamma$
  - Number of training epochs $K$
  - Batch size $B$
  - Timesteps per batch $T$

**Loop for each iteration**:

- Run policy $\pi_{\theta}$ in the environment and collect set of trajectories.
- Compute the rewards-to-go: $G_t = r_t + \gamma G_{t+1}$
- Compute the advantage estimates: $A_t = G_t - V_{\phi}(s_t)$
- For $K$ epochs, sample mini-batches of trajectories:
  - Compute ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$
  - Compute clipped surrogate objective: $L^{\text{clip}}(\theta) = 𝐄[\min (r_t(\theta) A_t, clip(r_t(\theta), 1 - \epsilon, 1+\epsilon)A_t)]$
  - Perform a gradient ascent step on $\theta$ using: $L^{\text{policy}}(\theta) = 𝐄[L^{\text{clip}}(\theta)]$
  - Minimize value loss: $L^{\text{value}}(\phi) = 𝐄[(G_t - V_{\phi}(s_t))^2]$
  - Perform a gradient descent step on $\phi$

## Notes

### Ratio

This ratio compares the probability of an action $a_t$ under the new policy $\pi_{\theta}$ with the old policy $\pi_{\theta_{\text{old}}}$. It tells us how much the new policy has changed relative to the old one for the same state-action pair.

$$
\begin{align*}
r_t(\theta) &= \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \\
\log r_t(\theta) &= \log \pi_{\theta}(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t) \\
r_t(\theta) &= \exp (\log \pi_{\theta}(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t))
\end{align*}
$$

### PPO-Clip

Policy gradient theorem:

$$
J(\theta) = 𝐄 [\log \pi_{\theta}(a_t | s_t) A_t]
$$

In PPO with ratio:

$$
J(\theta) = 𝐄 [r_t(\theta) (a_t | s_t) A_t]
$$

We limit its range to prevent extreme updates by clipping:

$$
clip(r_t(\theta), 1-\epsilon, 1+\epsilon)
$$

So,

$$
L^{\text{clip}}(\theta) = 𝐄[\min (r_t(\theta) A_t, clip(r_t(\theta), 1 - \epsilon, 1+\epsilon)A_t)]
$$
