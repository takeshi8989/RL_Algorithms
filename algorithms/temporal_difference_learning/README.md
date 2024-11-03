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

