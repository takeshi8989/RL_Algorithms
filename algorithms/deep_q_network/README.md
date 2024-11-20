# Deep Q-Network

## DQN

**Initialize**:
- learning rate $a$
- $\tau$
- number of batches per training step $B$
- number of updates per batch $U$
- batch size $N$
- experience replay memory with max size $K$
- Randomly initialize the network parameters $\theta$

**Loop for each steps**:
- Gather and store $h$ experiences ($s_i, a_i, r_i, s'_i$) using the current policy
- for $b = 1 ... B$:
    - sample a batch $b$ of experiences from the experience replay memory
    - for $u = 1 ... U$:
        - for $i = 1 ... N$:
            - $y_i = r_i + \delta_{s'_i} \gamma \max_{a_i} Q^{\pi_{\theta}} (s'_i, a'_i)$ where $\delta_{s'_i} = 0$ if $s'_i$ is terminal
        - $L(\theta) = \frac{1}{N} \sum_i {(y_i - Q^{\pi_{\theta}} (s'_i, a'_i))^2}$
        - $\theta = \theta - a \nabla_{\theta} L(\theta)$
- Decay $\tau$


## Notes

Softmax (Boltzmann) VS epsilon greedy