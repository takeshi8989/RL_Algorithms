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


