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
    - Compute clipped surrogate objective: $L^{\text{clip}}(\theta) = E[\min (r_t(\theta) A_t, clip(r_t(\theta), 1 - \epsilon, 1+\epsilon)A_t)]$
    - Add entropy bonus for exploration: $L^{\text{entropy}}(\theta) = \beta \cdot Entropy [\pi]$
    - Maximize total policy loss: $L^{\text{policy}}(\theta) = L^{\text{clip}}(\theta) + L^{\text{entropy}}(\theta)$
    - Update $\theta$ using gradient ascent.
    - Minimize value loss: $L^{\text{value}}(\phi) = E[(G_t - V_{\phi}(s_t))^s]$
    - Update $\phi$ using gradient ascent.

