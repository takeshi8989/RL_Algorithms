# Monte Carlo Methods (On-policy First-visit MC Control for Epsilon-soft policy)

## Pseudocode

**Inputs:**
- A set of states $S$
- A set of actions $A$
- Discount factor $\gamma$, where $0 \leq \gamma < 1$
- Exploration parameter $\epsilon$, where $0 \leq \epsilon < 1$

**Outputs:**
- Optimal action-value function $Q^*(s, a)$
- Improved policy $\pi^*$, which is $\epsilon$-soft

---

**Initialize**:
- $\pi(s) \gets$ arbitrarily $\epsilon$-soft policy
- $Q(s,a) \in R$ (arbitrarily), for all $s \in S$, $a \in A$
- $Returns(s,a) \gets$ empty list, for all $s \in S$, $a \in A$

**Loop for each episode**:
- Generate an episode based on $\pi$: $S_0, A_0, R_1, S_1, ... , S_{T-1}, A_{T-1}, R_T$
- $G \gets 0$
- Loop backward through each step $t$ in the episode ($t = T-1, T-2, ... , t=0$):
    - $G \gets \gamma G + R_{t+1}$
    - If $(s_t, a_t)$ is the first occurrence in the episode:
        - Append $G$ to $Returns(s_t, a_t)$
        - $Q(s_t, a_t) \gets avg(Returns(s_t, a_t))$
        - $A^* \gets \arg\max_a Q(s_t, a_t)$
        - For all $a \in A$:
            - If $a \neq A^*$: $\pi(a | S_t) \gets \epsilon / |A(S_t)|$
            - If $a = A^*$: $\pi(a | S_t) \gets 1 - \epsilon + \epsilon / |A(S_t)|$

**Return**:
- improved policy $\pi^*$ and action-value function $Q^*$

---

## Description
The $\epsilon$-soft policy ensures that the agent explores all actions in each state, balancing exploration and exploitation. By generating episodes, updating the action-value function based on observed returns, and adjusting the policy to be greedy with respect to $Q(s, a)$, this approach gradually improves $\pi$ toward the optimal policy $\pi^*$.

- Action-Value Function $Q(s,a)$: The expected $G$ of taking action $a$ in state $s$ and then following the policy $\pi$.
- Return $G(s,a)$: A list that stores the $G$ for each $(s, a)$ set.




## Notes

### Difference between First-visit MC and Every-visit MC

In the case Episode: $s_1 \rarr a_1 \rarr s_2 \rarr a_2 \rarr s_1 \rarr a_1 \rarr s_3$  ($s_1,a_1$ appears twice)

First-visit MC:
- Only uses the first $(s_1,a_1)$ occurrence to update $Q(s_1,a_1)$
- Return $G$ is calculated from first occurrence

Every-visit MC:
- Uses both $(s_1,a_1)$ occurrences
- Updates $Q(s_1,a_1)$ twice with different returns

#### Why use First-visit MC?
- Provides unbiased estimates of the expected value.
- Simpler theoretical analysis (returns are independent).
- Often preferred in practice due to these properties.


### Encouraging more exploration by using Epsilon-soft
- Without exploration ($\epsilon$ = 0), the agent might never try potentially better actions.
- Pure greedy policies ($\epsilon$ = 0) can get stuck in suboptimal solutions.
- This is different from Dynamic Programming where we can use a purely greedy policy because DP has complete knowledge of the environment.



<br/>
<br/>
<br/>
...
<br/>
<br/>



# Monte Carlo Methods (Off-policy MC Control with Weighted Importance Sampling)

## Pseudocode

**Inputs:**
- A set of states $S$
- A set of actions $A$
- Discount factor $\gamma$, where $0 \leq \gamma < 1$
- Target policy $\pi$ to be learned
- Behavior policy $b$ used to generate episodes

**Outputs:**
- Optimal action-value function $Q^*(s, a)$ for the target policy $\pi$

---

**Initialize**:
- $Q(s, a) \in R$ arbitrarily for all $s \in S$ and $a \in A$
- $C(s, a) \gets 0$ (cumulative weights) for all $s \in S$ and $a \in A$

**Loop for each episode**:
1. Generate an episode using behavior policy $b$: $S_0, A_0, R_1, S_1, \ldots, S_{T-1}, A_{T-1}, R_T$
2. $G \gets 0$
3. $W \gets 1$

4. **Loop backward through each step** $t$ **in the episode** ($t = T-1, T-2, \ldots, 0$):
    - $G \gets \gamma G + R_{t+1}$
    - $C(S_t, A_t) \gets C(S_t, A_t) + W$
    - $Q(S_t, A_t) \gets Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} \left( G - Q(S_t, A_t) \right)$
    - $\pi(S_t) \gets \arg\max_a Q(S_t, a)$
    - If $A_t \neq \pi(S_t)$, then break the loop
    - $W \gets W \cdot \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$

**Return**:
- Improved action-value function $Q^*$ for the target policy $\pi$

---


## Description
Off-policy Monte Carlo (MC) Control estimates $Q(s, a)$ while generating episodes using a behavior policy $b$. The target policy $\pi$ is the policy we aim to improve. This can be more exploratory than On-policy algorithm.

Off-policy MC relies on importance sampling to adjust for the differences between $\pi$ and $b$. Importance sampling weights correct the bias introduced by sampling from $b$ instead of $\pi$, ensuring that the value estimates converge accurately to those that would be obtained if we sampled directly from $\pi$.

- Importance sampling ratio $W$: adjusts the return $G$ in each episode to reflect the probabilities under the target policy $\pi$.
- Cumulative weights $C(s, a)$: tracks the cumulative sum of the importance sampling weights for each state-action pair across episodes.


## Notes

### Important Sampling Ratio

The probability of the subsequent state–action trajectory from ${S_t}$, occurring under any policy $\pi$:

$$
\begin{align*}
Pr(A_t,S_{t+1},A_{t+1},...,S_T | S_t,A_{t:T - 1} \sim \pi) 
&= \pi(A_t|S_t)p(S{t+1}|S_t,A_t)\pi(A_{t+1}|S_{t+1})···p(S_T |S_{T-1},A_{T-1}) \\
&= \prod_{k=t}^T \pi(A_k | S_k)p(S_{k+1}|S_k,A_k)
\end{align*}
$$

The important sampling ratio is:

$$
\begin{align*}
W_t &= \frac{\prod_{k=t}^{T-1} \pi(A_k | S_k)p(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1} b(A_k | S_k)p(S_{k+1}|S_k,A_k)} \\
  &=\prod_{k=t}^{T-1} \frac{\pi(A_k | S_k)}{b(A_k | S_k)} 
\end{align*}
$$


To estimate $V_\pi(s)$, we scale the returns by the ratios and average the results:

$$
V_n = \frac{\sum_{k=1}^{n-1} W_k G_k}{n}
$$

If you use **weighted** importance sampling ratio:

$$
V_n = \frac{\sum_{k=1}^{n-1} W_k G_k}{\sum_{k=1}^{n-1} W_k}
$$

If we define $C_n = \sum_{k=1}^{n} W_k$ as a cumulative sum of weights, $V_{n+1}$ is: 

```math
\begin{align*}
V_n     &= \frac{\sum_{k=1}^{n-1} W_k G_k}{C_{n-1}} \\
V_{n+1} &= \frac{\sum_{k=1}^{n} W_k G_k}{C_n} \\
        &= \frac{\sum_{k=1}^{n-1} W_k G_k + W_n G_n}{C_{n-1} + W_n} \\
        &= \frac{C_{n-1}V_n + W_n G_n}{C_{n-1} + W_n} \\
        &= \frac{C_{n-1}}{C_{n-1} + W_n}V_n + \frac{W_n}{C_{n-1} + W_n}G_n \\
        &= V_n(\frac{C_{n-1}}{C_n}) + \frac{W_n}{C_n}G_n \\
        &= V_n(1 - \frac{W_n}{C_n}) + \frac{W_n}{C_n}G_n \\
        &= V_n + \frac{W_n}{C_n}G_n - \frac{W_n}{C_n}V_n \\
        &= V_n + \frac{W_n}{C_n}(G_n - V_n)
\end{align*}
```


### Ordinary IS vs Weighted IS

Regular IS:     
$$
Q(s,a) = sum(W_i * G_i) / n
$$

Weighted IS:
$$
\begin{align*}
Q(s,a) &= sum(W_i * G_i) / sum(W_i) \\
       &= sum(W_i * G_i) / C(s,a)
\end{align*}
$$


- Regular IS can have high variance
- $C(s,a)$ in the denominator helps normalize the update
- Thus, Weighted IS produces lower variance estimates
- And Weighted IS is guaranteed to be bounded by the maximum possible value