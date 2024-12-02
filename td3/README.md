# Twin Delayed DDPG (TD3)

**Initialize**:
- Actor (policy) network: $\mu_{\theta}(s)$ with parameters $\theta$
- Two Q-function (critic) networks: $Q_{\phi_1}(s, a), Q_{\phi_2}(s, a)$ with parameters $\phi_1, \phi_2$
- Target networks: $\mu_{\text{target}}, Q_{\text{target,1}}, Q_{\text{target,2}}$, initialized to match their respective main networks
- Empty replay buffer $\mathcal{D}$
- Hyperparameters:
    - Exploration noise $\epsilon \sim \mathcal{N}(0, \sigma)$
    - Target policy noise $\mathcal{N}(0, c)$
    - Update delay $\text{policy\_delay}$
    - Learning rates $\alpha_{\text{actor}}, \alpha_{\text{critic}}$

---

**Loop for each iteration**:

- For each timestep:
    - Observe state $s$ and select action $a = \text{clip}(\mu_{\theta}(s) + \epsilon, a_{\text{low}}, a_{\text{high}})$
    - Execute action $a$ in the environment, observe reward $r$, next state $s'$, and terminal signal $d$
    - Store $(s, a, r, s', d)$ in replay buffer $\mathcal{D}$

- Sample batch $B = \{(s, a, r, s', d)\}$ from $\mathcal{D}$
- Compute Target Actions: $a' = \text{clip}(\mu_{\text{target}}(s') + \text{clip}(\mathcal{N}(0, c), -c, c), a_{\text{low}}, a_{\text{high}})$
- Compute Target Q-Values: $y(r, s', d) = r + \gamma (1 - d) \min\limits_{i=1,2} Q_{\text{target, i}}(s', a')$
- Update Critic Networks: $\mathcal{L}_{\phi_i} = \frac{1}{|B|} \sum\limits_{(s, a, r, s', d) \in B} \left( Q_{\phi_i}(s, a) - y(r, s', d) \right)^2$
- If $j \mod \text{policy\_delay} = 0$, update actor parameters $\theta$:
    - $\mathcal{L}_{\theta} = -\frac{1}{|B|} \sum\limits_{s \in B} Q_{\phi_1}(s, \mu_{\theta}(s))$
- Update Target Networks: 
    - $\phi_{\text{target, i}} \gets \tau \phi_i + (1 - \tau) \phi_{\text{target, i}}$ 
    - $\theta_{\text{target}} \gets \tau \theta + (1 - \tau) \theta_{\text{target}}$

