# Soft Actor-Critic (SAC)

**Initialize**:
- Actor (policy): Stochastic policy network $\pi_{\theta}(a|s)$ with parameters $\theta$, outputs a Gaussian distribution.
- Critic networks: Two Q-function networks  $Q_{\phi_1}(s, a), Q_{\phi_2}(s, a)$ with parameters $\phi_1, \phi_2$
- Value network: $V_{\psi}(s)$ with parameters $\psi$
- Target network for critics: $Q_{\text{target,1}}, Q_{\text{target,2}}$, initialized to match the main networks.
- Entropy coefficient $\alpha$
- Empty replay buffer $\mathcal{D}$
- Hyperparameters:
    - Discount factor $\gamma$
    - Target smoothing coefficient $\tau$
    - Learning rates $\alpha_{\text{actor}}, \alpha_{\text{critic}}, \alpha_{\text{value}}$
    - Temperature parameter $\alpha$
---

**Loop for each iteration**:

- For each timestep:
    - Observe state $s$ and select action $a \sim \pi_{\theta}(a|s)$
    - Execute action $a$ in the environment, observe reward $r$, next state $s'$, and terminal signal $d$
    - Store $(s, a, r, s', d)$ in replay buffer $\mathcal{D}$

- Sample batch $B = \{(s, a, r, s', d)\}$ from $\mathcal{D}$
- Compute Target Actions: $a' \sim \pi_{\theta}(a'|s')$
- Compute Target Q-Values: $y(r, s', d) = r + \gamma (1 - d) (\min\limits_{i=1,2} Q_{\text{target, i}}(s', a') - \alpha \log \pi_{\theta}(a'|s'))$
- Update Critic Networks: $\mathcal{L}_{\phi_i} = \frac{1}{|B|} \sum\limits_{(s, a, r, s', d) \in B} \left( Q_{\phi_i}(s, a) - y(r, s', d) \right)^2$
- Maximize the expected reward and entropy: $\mathcal{L}_{\theta} = -\frac{1}{|B|} \sum\limits_{s \in B} (\alpha \log \pi_{\theta}(a|s) - \min\limits_{i=1,2}Q_{\phi_i}(s, a))$
- Update Target Networks: $\phi_{\text{target, i}} \gets \tau \phi_i + (1 - \tau) \phi_{\text{target, i}}$ 


## Notes

### Key Differences: SAC vs TD3
1. Policy Type:
    - TD3: Deterministic policy; directly outputs actions.
        - $a = \pi_{\theta}(s)$
    - SAC: Stochastic policy; outputs a probability distribution and samples actions. 
        - $a \sim \pi_{\theta}(a|s)$

2. Exploration Mechanism:
    - TD3: Adds noise to the deterministic policy during training.
        - $a = clip(\pi_{\theta}(s) + \epsilon, a_{\text{low}}, a_{\text{high}})$
    - SAC: Uses entropy regularization to encourage exploration.
        - $\mathcal{L}_{\text{actor}} = E[\alpha \log \pi(a|s) - Q(s, a)]$

3. Critic Updates:
    - TD3: Minimizes overestimation bias with clipped double Q-learning.
        - $y(r, s', d) = r + \gamma (1 - d) \min (Q_{\phi_1}, Q_{\phi_2})$
    - SAC: Similar but includes an entropy term for exploration.
        - $y(r, s', d) = r + \gamma (1 - d)(\min (Q_{\phi_1}, Q_{\phi_2}) - \alpha \log \pi_{\theta}(a|s))$
        
