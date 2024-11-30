# Trust Region Policy Optimization (TRPO)

**Initialize**:

- Policy network $\pi_{\theta}(a|s)$ with parameters $\theta$
- Value function $V_{\phi}(s)$ with parameters $\phi$
- Hyperparameters
  - KL-divergence limit $\delta$
  - Backtracking coefficient $\alpha$
  - Maximum number of backtracking steps $K$

**Loop for each iteration**:
- Run policy $\pi_{\theta}$ in the environment and collect set of trajectories.
- Compute the rewards-to-go: $G_t = r_t + \gamma G_{t+1}$
- Compute the advantage estimates: $A_t = G_t - V_{\phi}(s_t)$
- Compute the policy gradient: $\hat{g_k} = \frac{1}{|D_k|} \sum_{\tau \in D_k} \sum_{t=0}^T \nabla \log \pi_{\theta}(a_t|s_t) |_{\theta_k} A_t$
- Use the conjugate gradient method to compute search direction: $\hat{x_k} = \hat{H_k}^{-1} \hat{g_k}$
- Backtracking line search to update the policy: $\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2\delta}{{\hat{x_k}^T} \hat{H_k} \hat{x_k}}} \hat{x_k}$
- Update Value Function: $\phi_{k+1} = \arg \min\limits_{\phi} \frac {1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T (V_{\phi}(s_t) - \hat{R_t})^2$



## Notes


### Conjugate Gradient Method

TRPO Optimization Problem:

$$
\max\limits_{\theta} E_{\pi_{\theta_{\text{old}}}} [\hat{A}(s, a)],  \text{ subject to: } D_{\text{KL}} (\pi_{\theta_{\text{old}}} || \pi_{\theta}) \leq \delta
$$

Where:
- $E_{\pi_{\theta_{\text{old}}}} [\hat{A}(s, a)]$: Expected advantage under the new policy.
- $D_{\text{KL}} (\pi_{\theta_{\text{old}}} || \pi_{\theta})$: KL divergence
- $\delta$: The allowed KL divergence threshold (Trust Region size).


The objective is rewritten in terms of a small update $x$, where $x$ represents a step in the parameter space of the policy:

$$
\theta_{\text{new}} = \theta_{\text{old}} + x
$$

Using a first-order Taylor approximation, the expected advantage can be written as:

$$
E_{\pi_{\theta_{\text{old}}}} [\hat{A}(s, a)] \approx g^T x
$$

Where:
- $g^T$: Gradient of the objective with respect to the policy parameters $\theta$
- $x$: The update direction in the parameter space.

So, the objective can be written as:

$$
\max\limits_{x} g^T x
$$

We approximate the KL divergence with a second-order Taylor expansion to make it computationally manageable:

$$
D_{\text{KL}} (\pi_{\theta_{\text{old}}} || \pi_{\theta}) \approx \frac{1}{2} x^T Hx
$$

Where:
- $H$: The Fisher information matrix, which measures the curvature of the KL divergence with respect to the policy parameters.

So, The TRPO optimization problem is reformulated as:

$$
\max\limits_{x} g^T x,  \text{ subject to: } x^T Hx \leq \delta
$$

- We absorb the factor of $\frac{1}{2}$ into $\delta$
- And this is a quadratic programming problem

### Line Search

We use the method of Lagrange multipliers to solve this constrained optimization problem.

$$
\begin{align*}
L(x, \lambda) &= g^T x - \lambda (x^T Hx - \delta) \\
\nabla_x L(x, \lambda) &= g - 2 \lambda Hx = 0 \\
Hx &= \frac {g}{2 \lambda} \\
x &= \frac {1}{2 \lambda} H^{-1} g
\end{align*}
$$

Where:
- $\lambda \geq 0$: The Lagrange multiplier.

And then, substitute $x = \frac {1}{2 \lambda} H^{-1}g$ into the constraint $x^T Hx \leq \delta$:

$$
\begin{align*}
(\frac {1}{2 \lambda} H^{-1}g)^T H (\frac {1}{2 \lambda} H^{-1}g) &\leq \delta \\
\frac {1}{(2 \lambda)^2} g^T H^{-1} H H^{-1}g &\leq \delta \\
\frac {1}{(2 \lambda)^2} g^T H^{-1}g &\leq \delta \\
g^T H^{-1}g &\leq (2 \lambda)^2 \delta \\
\lambda &= \sqrt{\frac{g^T H^{-1} g}{4 \delta}}
\end{align*}
$$

Substitute $\lambda = \sqrt{\frac{g^T H^{-1} g}{4 \delta}}$ back into $x = \frac {1}{2 \lambda} H^{-1}g$:

$$
\begin{align*}
x &= \frac {1}{2 \sqrt{\frac{g^T H^{-1} g}{4 \delta}}} H^{-1}g \\
&= \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g \\
x_{\text{scaled}}&= \sqrt{\frac{2 \delta}{x^T Hx}} x
\end{align*}
$$