# Temporal Difference Learning - TD(0)

## Tabular TD(0)

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets \pi(S)$
    - Take action $A$, observe $R, S'$
    - $V(S) \gets V(S) + a [R + \gamma V(S') - V(S)]$
    - $S \gets S'$


## SARSA

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets$ action derived from $Q$ (e.g $\epsilon$-greedy)
    - Take action $A$, observe $R, S'$
    - $A' \gets$ action derived from $Q(S', A)$
    - $Q(S, A) \gets Q(S, A) + a [R + \gamma Q(S', A') - Q(S, A)]$
    - $S \gets S', A \gets A'$



## Expected SARSA

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets$ action derived from $Q$ (e.g $\epsilon$-greedy)
    - Take action $A$, observe $R, S'$
    - $Q(S, A) \gets Q(S, A) + a [R + \gamma \sum_{a'} \pi(a' | S') Q(S', a') - Q(S, A)]$
    - $S \gets S'$



## Q-Learning

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets$ action derived from $Q$ (e.g $\epsilon$-greedy)
    - Take action $A$, observe $R, S'$
    - $Q(S, A) \gets Q(S, A) + a [R + \gamma \max_a Q(S', a) - Q(S, A)]$
    - $S \gets S'$


## Double Q-Learning

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets$ action from $\epsilon$-greedy policy in $Q_1$ and $Q_2$
    - Take action $A$, observe $R, S'$
    - With 0.5 probability:
      - $Q_1(S, A) \gets Q_1(S, A) + a [R + \gamma Q_2(S', \arg\max_a Q_1(S', a)) - Q_1(S, A)]$
    - else:
      - $Q_2(S, A) \gets Q_2(S, A) + a [R + \gamma Q_1(S', \arg\max_a Q_2(S', a)) - Q_2(S, A)]$
    - $S \gets S'$



## Description
Temporal Difference (TD) Learning combines ideas from Monte Carlo methods and dynamic programming. TD updates estimates based on current approximations of other estimates.

$$
V(S) \gets V(S) + a [R + \gamma V(S)]
$$


- SARSA: Extends TD(0) to action-value functions in an on-policy setting.
- Expected SARSA: Uses the expectation over the next actions for updating the value.
- Q-Learning: An off-policy variant where updates are based on the maximum estimated future value.
- Double Q-Learning: Addresses overestimation issues in Q-learning by maintaining two independent Q-value estimates.


<br/><br/><br/>

# n-step TD


## n-step TD Prediction

**Loop for each episode**:
- Initialize $S_0 \neq terminal$
- $T = \infty$
- Loop for $t = 0, 1, 2, ...$:
    - If $t < T$:
        - $A_t \gets \pi(S_t)$
        - Observe and Store $R_{t+1}, S_{t+1}$
        - If $S_{t+1} = S_T$: 
            - $T \gets t+1$
    - $\tau \gets t-n+1$
    - If $\tau \geq 0$:
        - $G \gets \sum_{i=\tau+1}^{min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
        - If $\tau+n \lt T$: 
            - $G \gets G + \gamma^n V(S_{\tau+n})$
        - $V(S_{\tau}) \gets V(S_{\tau}) + a [G - V(S_{\tau})]$
- Until $\tau = T-1$


## n-step Sarsa

**Loop for each episode**:
- Initialize $S_0 \neq terminal$
- Select and store $A_0 \gets \pi(S_0)$
- $T = \infty$
- Loop for $t = 0, 1, 2, ...$:
    - If $t < T$:
        - $A_t \gets \pi(S_t)$
        - Observe and Store $R_{t+1}, S_{t+1}$
        - If $S_{t+1} = S_T$: 
            - $T \gets t+1$
        - else: 
            - $A_{t+1} \gets \pi(S_{t+1})$
    - $\tau \gets t-n+1$
    - If $\tau \geq 0$:
        - $G \gets \sum_{i=\tau+1}^{min(\tau+n, T)} \gamma^{i-\tau-1} R_i$
        - If $\tau+n \lt T$: 
            - $G \gets G + \gamma^n Q(S_{\tau+n}, A_{\tau+n})$
        - $Q(S_{\tau}, A_{\tau}) \gets Q(S_{\tau}, A_{\tau} + a [G - Q(S_{\tau}, A_{\tau})]$

    - If $\pi$ is being learned, then ensure that $\pi(S_{\tau})$ is $\epsilon$-greedy with respect to $Q$
- Until $\tau = T-1$

