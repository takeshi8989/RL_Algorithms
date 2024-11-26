# Advantage Actor-Critic (A2C)

**Initialize**:
- Policy parameters $\theta$ and value function parameters $\phi$ arbitrarily
- Learning rate $a_{\theta}$ and $a_{\phi}$

**Loop for each episode**:
- Initialize state $s_0$
- Loop for each stap $t = 0, 1, 2 ...$:
    - Sample action $a_t \sim \pi_{\theta} (a_t | s_t)$
    - Execute action $a_t$, observe reward $r_{t+1}$ and next state $s_{t+1}$
    - Compute advantage: $A_t = r_{t+1} + \gamma V(s_{t+1}; \phi) - V(s_t; \phi)$
    - Update policy parameters: $\theta \gets \theta + a \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) A_t$
    - Update value function parameters: $\phi \gets \phi - a_{\phi} \nabla_{\phi} (r_{t+1} + \gamma V(s_{t+1}; \phi) - V(s_t; \phi))^2$
    - If $s_{t+1}$ is terminal: break


## Notes

### The Avarage

$$
\begin{align*}
A^{\pi}(s_t, a_t) &= Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) \\
&= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... + \gamma^n r_{t+n} + \gamma^{n+1} V^{\pi}(s_{t+n+1}) - V^{\pi}(s_t) \\
&= r_{t+1} + \gamma V(s_{t+1}; \phi) - V(s_t; \phi)
\end{align*}
$$

