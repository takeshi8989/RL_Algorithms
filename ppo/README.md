# Proximal Policy Optimization (PPO)

**Initialize**:
- Policy network $\pi_{\theta}(a|s)$ with parameters $\theta$
- Value function $V_{\phi}(s)$ with parameters $\phi$
- Clipping parameter $\epsilon$, learning rates $a_{\theta}$ and $a_{\phi}$, discount factor $\gamma$, and entropy coefficient $\beta$

**Loop for each iteration**:
- Run policy $\pi_{\theta}$ in the environment for $T$ timesteps, store $s_t, a_t, r_t$, and $\log \pi_{\theta}(a|s)$ 
- Loop for $t = T-1, T-2, ..., 0$:
    - Compute the return $G_t = r_t + \gamma G_{t+1}$
    - Compute the advantage $A_t = G_t - V_{\phi}(s_t)$
- For $K$ epochs, , sample mini-batches of trajectories:
    - Compute ratio: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$
    - Compute clipped surrogate objective: $L^{\text{clip}}(\theta) = E[\min (r_t(\theta) A_t, clip(r_t(\theta), 1 - \epsilon, 1+\epsilon)A_t)]$
    - Add entropy bonus for exploration: $L^{\text{entropy}}(\theta) = \beta \cdot Entropy [\pi]$
    - Maximize total policy loss: $L^{\text{policy}}(\theta) = L^{\text{clip}}(\theta) + L^{\text{entropy}}(\theta)$
    - Update $\theta$ using gradient ascent.
- Minimize value loss: $L^{\text{value}}(\phi) = E[(G_t - V_{\phi}(s_t))^s]$
- Update $\phi$ using gradient ascent.

