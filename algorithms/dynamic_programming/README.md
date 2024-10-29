# Dynamic Programming

## Pseudocode

**Policy Iteration**

**Inputs:**
- A set of states $S$
- A set of actions $A$
- Transition probabilities $P(s' | s, a)$
- Rewards $R(s, a)$
- Discount factor $\gamma$, where $0 \leq \gamma < 1$

**Outputs:**
- Optimal policy $\pi^*$
- Optimal state-value function $V^*$

---

1. **Initialize** policy $\pi(s)$ arbitrarily for each state $s \in S$
2. **Initialize** $V(s) = 0$ for each state $s \in S$

3. **Loop until policy converges**:
   - **Policy Evaluation**: Evaluate $\pi$ to obtain $V(s)$
     - Loop until values converge:
       - For each state $s \in S$:
         - $V(s) \gets \sum_{a \in A} \pi(a | s) \sum_{s' \in S} P(s' | s, a) \left[ R(s, a) + \gamma V(s') \right]$

   - **Policy Improvement**: Improve $\pi$ based on $V(s)$
     - For each state $s \in S$:
       - $\pi(s) \gets \arg\max_a \sum_{s' \in S} P(s' | s, a) \left[ R(s, a) + \gamma V(s') \right]$

4. **Return** optimal policy $`\pi^*`$ and state-value function $`V^*`$


---

### Description
The goal of the algorithm is to iteratively improve the policy $\pi$ to find the optimal policy $\pi^*$, which maximizes the cumulative reward. This is achieved by repeatedly evaluating the current policy's state-value function $V(s)$ and updating the policy to make it greedy with respect to $V(s)$.

- **States $S$**: A finite set of states representing possible situations in an environment.
- **Actions $A$**: A set of possible actions an agent can take.
- **Policy $\pi(a|s)$**: A probability distribution over actions given each state, indicating which action to take.
- **Transition Probabilities $P(s' | s, a)$**: The probability of moving from state $s$ to $s'$ after taking action $a$.
- **Rewards $R(s, a)$**: The immediate reward received after transitioning from state $s$ to $s'$ via action $a$.
- **Discount Factor $\gamma$**: A factor $0 \leq \gamma < 1$ used to weigh future rewards, ensuring that the sum of rewards remains finite.


## Notes

### Policy Evaluation
**Objective: Evaluate the current policy to determine the expected value of each state under that policy.**

The expected return $G$ is computed based on the immediate reward and the discounted future rewards. \
The optimal State-Value Function $`v^*(s)`$ and the action-value function $`q^*(s, a)`$ can be expressed as:


```math
\begin{align*}
v^*(s) &= \max_a E[G_t | S_t=s, A_t=a] \\
       &= \max_a E[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a] \\
       &= \max_a E[R_{t+1} + \gamma v^*(S_{t+1}) | S_t=s, A_t=a] \\ 
       &= \max_a \sum_{s', r} P(s', r| s, a) \left[ r + \gamma v^*(s') \right]
\end{align*}
```


and 


```math
\begin{align*}
q^*(s, a) &= E[G_t | S_t=s, A_t=a] \\
          &= E[R_{t+1} + \gamma \max_{a'} q^*(S_{t+1}, a') | S_t=s, A_t=a] \\
          &= \sum_{s', r} P(s', r| s, a) \left[ r + \gamma \max_{a'} q^*(s', a') \right]
\end{align*}
```


Based on the policy $\pi$, the value functions look like:

$$
\begin{align*}
v_{\pi}(s) &= \sum_{a} \pi(a | s) \sum_{s', r} P(s', r| s, a) \left[ r + \gamma v_{\pi}(s') \right] \\
q_{\pi}(s, a) &= \sum_{s', r} P(s', r| s, a) \left[ r + \gamma v_{\pi}(s') \right]
\end{align*}
$$

We use them to update our policy:

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s' | s, a) \left[ R(s, a) + \gamma v_{\pi}(s') \right] 
$$



### Policy Improvement
**Objective: Refine the current policy $\pi$ to maximize expected returns based on the value function $V(s)$.**

Since 

$$
\begin{align*}
q_{\pi}(s,\pi'(s)) &= \sum_{s', r} P(s', r| s, \pi'(s)) \left[ r + \gamma v_{\pi}(s') \right] \\
v_{\pi}(s) &= \sum_a \pi(a | s) q_{\pi}(s, a) = \sum_a \pi(a | s) \sum_{s', r} P(s', r| s, a) \left[ r + \gamma v_{\pi}(s') \right]
\end{align*}
$$

This is guaranteed ( $\pi'(s) = \max(A) \geq \text{any } a$ ):

$$
\begin{align*}
q_{\pi}(s,\pi'(s)) & \geq \sum_a \pi(a | s) q_{\pi}(s, a) \\
q_{\pi}(s,\pi'(s)) & \geq v_{\pi}(s) 
\end{align*}
$$

Also:

$$
\begin{align*}
v_{\pi'}(s) &= \sum_{s', r} P(s', r| s, \pi'(s)) \left[ r + \gamma v_{\pi}(s') \right] \\
            &= q_{\pi}(s,\pi'(s)) \\
            & \geq v_{\pi}(s) 
\end{align*}
$$


### Policy Iteration
**Objective: Iteratively find the optimal policy $\pi^*$ by alternating between evaluating the current policy and improving it.**

$$
\pi_0 \xrightarrow{E} v_{\pi_{0}} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_{1}} \xrightarrow{I} \pi_2 \xrightarrow{E} ... \xrightarrow{I} \pi_* \xrightarrow{E} v_{\pi_{*}} 
$$
