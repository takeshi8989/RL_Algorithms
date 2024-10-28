# Multi-Armed Bandit (Epsilon Greedy)

## Pseudocode

Initialize, for a = 1 to k:
  - Q(a) &larr; 0
  - N(a) &larr; 0

Loop for the number of iterations:
  1. Choose action A as follows:
     - $\arg\max_a Q(a)$ with probability 1 - $\epsilon$
     - a random action with probability $\epsilon$
  2. Observe the reward
     - R &larr; bandit(A)
  3. Update:
     - N(A) &larr; N(A) + 1
     - Q(A) &larr; Q(A) + $\frac{1}{N(A)} \left[ R - Q(A) \right]$

---

### Description

- You can take one of k actions regardless of the state.
- Q(a) is the estimated value of action.
- N(a) is the number of times the action a has been selected.
- $\epsilon$ is a small probability that chooses a random action.
- bandit(a) is a fuction that returns a reward based on the action.
- R is a reward given by bandit(a)

---

### Notes
$q^*(a)$ is is the expected reward given that $a$ is selected. 

$$
q^*(a) = E[R_t \mid A_t = a]
$$

$Q_t(a)$ is the estimated the estimated value of action a at time step $t$

$$
Q_t(a) = \frac{\text{sum of rewards when $a$ taken prior to t}}{\text{number of times $a$ taken prior to t}} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}\{A_i = a\}}{\sum_{i=1}^{t-1} \mathbb{I}\{A_i = a\}}
$$

When you take a greedy action $A_t$, the algorithm is:

$$
A_t = \arg\max_a Q_t(a)
$$

And $Q_{n+1}$ is

$$
\begin{align*}
Q_{n+1} &= \frac{R_1 + R_2 + \cdots + R_n}{n} \\
        &= \frac{1}{n} \left( \sum_{i=1}^{n} R_i \right) \\
        &= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
        &= \frac{1}{n} \left( R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right) \\
        &= \frac{1}{n} \left( R_n + (n-1) Q_n \right) \\
        &= \frac{1}{n} \left( R_n + n Q_n - Q_n \right) \\
        &= Q_n + \frac{1}{n} ( R_n - Q_n) \\
\end{align*}
$$
