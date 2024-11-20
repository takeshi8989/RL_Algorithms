# Deep Q-Network

**Initialize**:
- Replay memory $M$ with a fixed capacity
- Action-value function $Q$ with random weights
- Target network $Q_{target}$ with the same structure as $Q$
- Discount factor $\gamma$, learning rate $a$, batch size, and exploration parameter $\epsilon$

**Loop for each episode**:
- Reset the environment and get the initial state $s$
- Repeat for each time step
    - Gather and store experiences $(s, a, r, s')$ using the current policy
    - Sample a random mini-batch of transitions $(s_i, a_i, r_i, s'_i)$ from $M$
    - Computer the target: $y = r_i + \gamma \max_{a'} Q_{target} (s'_i, a')$ 
    - Update the $Q$ by minimizeing the loss: $L = \frac{1}{N} \sum_i {(y_i - Q(s_i, a_i))^2}$
    - Periodically update: $Q_{target} \gets Q$

## Notes

### Replay Memory
Replay memory is a data structure (usually a buffer) that stores past experiences (transitions) of the form $(s, a, r, s')$. Instead of updating the network immediately with the latest experience, experiences are sampled randomly from the buffer for training.
- Random sampling ensures the training data is more independent and identically distributed.
- Experiences can be reused multiple times, improving sample efficiency.

### Periodical Update
In Deep Q-Networks, the target network $Q_{target}$ is updated less frequently (e.g., every few episodes or steps) than the main network $Q$.
- The target $y$ become more stable since $Q_{target}$ doesn't change at every step.
- This prevents rapid oscillations in learning caused by bootstrapping updates.

### Exploration Strategy

1. Epsilon Greedy

$$
\begin{align*}
a &= \text{random action with probability } \epsilon \\
a &= \arg\max_a'Q(s, a') \text{with probability } 1 - \epsilon
\end{align*}
$$



2. Softmax (Boltzmann Exploration)

    - Select actions probabilistically based on their Q-values using a softmax distribution.
    - Balances exploration and exploitation more intelligently than epsilon-greedy by favoring actions with higher Q-values.

$$
P(a|s) = \frac{e^{\frac{Q(s,a)}{\tau}}}{\sum_{a'} e^{\frac{Q(s,a')}{\tau}}}
$$
