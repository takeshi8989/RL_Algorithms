# REINFORCE

**Initialize**:
- Policy parameters $\theta$ arbitrarily

**Loop for each episode**:
- Generate an episode $(s_0, a_0, r_1, s_1, ..., s_{T-1}, a_{T-1}, r_T) \sim \pi_{\theta}$
- for $t = 0$ to $T-1$:
    - Compute return $V_t = \sum_{k=t}^T \gamma^{k-t} r_{k+1}$
    - $\nabla_{\theta} J(\pi_{\theta}) \gets \nabla_{\theta} J(\pi_{\theta}) + \nabla_{\theta} \log \pi_{\theta} (a_t | s_t) V_t$
- $\theta \gets \theta + a \nabla_{\theta} J(\pi_{\theta})$

## Note

### Policy Gradient

- $a$: The learning rate (step size for gradient ascent).
- $\nabla_{\theta} \log \pi_{\theta} (a_t | s_t)$: The gradient of the log-probability of taking action $a_t$ in state $s_t$ with respect to the policy parameters $\theta$.
- $V_t$: The return at time $t$, representing the total discounted reward from time $t$ onwards.

### Why we use the log-probability

``` math
\begin{align*}
\max_{\theta} J(\theta) &= \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)] \\
\nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)] \\
&= \nabla_{\theta} \sum_{\tau} p(\tau | \theta) R(\tau) \\
&= \sum_{\tau} \nabla_{\theta} (p(\tau | \theta) R(\tau)) \\
&= \sum_{\tau} (p(\tau | \theta) \nabla_{\theta} R(\tau) + R(\tau)  \nabla_{\theta} p(\tau | \theta))  \\
&= \sum_{\tau} R(\tau) \nabla_{\theta} p(\tau | \theta)  \\
&= \sum_{\tau} R(\tau) p(\tau | \theta) \frac{\nabla_{\theta} p(\tau | \theta)}{p(\tau | \theta)}   \\
&= \sum_{\tau} p(\tau | \theta) R(\tau) \nabla_{\theta} \log p(\tau | \theta)  \\
&= \mathbb{E}_{\tau \sim \pi_{\theta}} [\sum_{t=0}^T R_t (\tau) \nabla_{\theta} \log p(\tau | \theta)]
\end{align*}
```

And

``` math
\begin{align*}
p(\tau | \theta) &= \prod_{t \geq 0} p(s_{t+1} | s_t, a_t) \pi_{\theta} (a_t | s_t) \\
\log p(\tau | \theta) &= \log \prod_{t \geq 0} p(s_{t+1} | s_t, a_t) \pi_{\theta} (a_t | s_t) \\
&= \sum_{t \geq 0} \log p(s_{t+1} | s_t, a_t) + \log \pi_{\theta} (a_t | s_t) \\
\nabla_{\theta} \log p(\tau | \theta) &= \nabla_{\theta} \sum_{t \geq 0} \log p(s_{t+1} | s_t, a_t) + \log \pi_{\theta} (a_t | s_t) \\
&= \nabla_{\theta} \sum_{t \geq 0} \log \pi_{\theta} (a_t | s_t)
\end{align*}
```

Thus, 

``` math
\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} [\sum_{t=0}^T R_t (\tau) \nabla_{\theta} \log \pi_{\theta} (a_t | s_t)]
```
