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
- Use the conjugate gradient method to compute search direction: $\hat{x_k} = \hat{H_k}^{-1} \cdot \hat{g_k}$
- Backtracking line search to update the policy: $\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2\delta}{{\hat{x_k}^T} \hat{H_k} \hat{x_k}} \hat{x_k}}$
- Update Value Function: $\phi_{k+1} = \arg \min\limits_{\phi} \frac {1}{|D_k|T} \sum_{\tau \in D_k} \sum_{t=0}^T (V_{\phi}(s_t) - \hat{R_t})^2$