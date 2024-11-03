# Temporal Difference Learning - TD(0)

## Tabular TD(0)

**Loop for each episode**:
- Initialize $S$
- Loop for each step of episode:
    - $A \gets \pi(S)$
    - Take action $A$, observe $R, S'$
    - $V(S) \gets V(S) + a [R + \gamma V(S') - V(S)]$
    - $S \gets S'$
