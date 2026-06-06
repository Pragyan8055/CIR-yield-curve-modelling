# Stochastic Interest Rate Modelling and Prediction
### Implementing, Calibrating, and Extending the Cox–Ingersoll–Ross Model on Real Yield Curve Data
**Finance Club, IIT Roorkee — Open Projects 2026**

---

## Project Overview

This project implements a complete analytical pipeline for the Cox–Ingersoll–Ross (CIR) short-rate model applied to historical zero-coupon yield curve data spanning nine maturities (3M through 30Y). Three interlocking analytical layers build on one another mathematically: stochastic calculus foundations, financial econometrics, and statistical learning. The prediction task is strict: reconstruct the entire yield curve using **only the 3M yield** as input on each test date.

---

## Repository Structure

```
.
├── notebook1_CIR_analytical.ipynb    # Analytical research notebook
├── notebook2_modular_OOP.ipynb       # Modular OOP + DSA implementation
├── cir_full_report.pdf               # Full project report (36 pages)
├── train_data.csv                    # Training yield curve data
├── test_data.csv                     # Test yield curve data (actuals)
├── test_data_3M.csv                  # Test 3M yield (only permitted input)
└── images/                           # All generated figures
```

---

## Mathematical Framework

### 1. The CIR Stochastic Differential Equation

The Cox–Ingersoll–Ross model describes the instantaneous short rate $r_t$ via:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t, \qquad r_0 > 0$$

| Parameter | Symbol | Interpretation |
|-----------|--------|----------------|
| Speed of mean reversion | $\kappa > 0$ | How fast $r_t$ returns to $\theta$ |
| Long-run mean | $\theta > 0$ | Equilibrium interest rate |
| Volatility | $\sigma > 0$ | Magnitude of random fluctuations |
| Brownian motion | $W_t$ | Standard Wiener process |

**Feller Condition** — guarantees $r_t > 0$ almost surely:

$$2\kappa\theta \geq \sigma^2$$

**Calibrated parameters** (MCMC posterior medians):

$$\hat\kappa = 0.0474, \quad \hat\theta = 4.29\%, \quad \hat\sigma = 0.0425$$

Mean-reversion half-life: $t_{1/2} = \log 2 / \kappa \approx 14.6$ years (3,683 trading days).

---

### 2. Closed-Form Moments via Itô's Lemma

Conditional mean (applying Itô's lemma to $f(r_t) = r_t$):

$$\mathbb{E}[r_t \mid r_0] = \theta + (r_0 - \theta)\,e^{-\kappa t}$$

Conditional variance (applying Itô's lemma to $r_t^2$):

$$\text{Var}(r_t \mid r_0) = \frac{\sigma^2\theta}{2\kappa}(1-e^{-\kappa t})^2 + \frac{\sigma^2 r_0}{\kappa}\,e^{-\kappa t}(1-e^{-\kappa t})$$

Stationary distribution as $t \to \infty$:

$$r_\infty \sim \text{Gamma}\!\left(\frac{2\kappa\theta}{\sigma^2},\;\frac{\sigma^2}{2\kappa}\right)$$

---

### 3. Exact Transition Density (Non-Central Chi-Squared)

$$p(r_t \mid r_s) = c\,e^{-u-v}\!\left(\frac{v}{u}\right)^{q/2}\!I_q(2\sqrt{uv})$$

$$c = \frac{2\kappa}{\sigma^2(1-e^{-\kappa\Delta t})}, \quad u = c\,r_s e^{-\kappa\Delta t}, \quad v = c\,r_t, \quad q = \frac{2\kappa\theta}{\sigma^2}-1$$

Equivalently: $2c\,r_t \mid r_s \sim \chi^2(2q+2,\;2u)$ — a non-central chi-squared distribution. This exact density drives the MCMC log-likelihood, avoiding all Euler-discretisation bias.

---

### 4. Fokker–Planck Equation (Kolmogorov Forward)

The evolution of the probability density $p(r,t)$:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial r}\bigl[\kappa(\theta-r)\,p\bigr] + \frac{1}{2}\frac{\partial^2}{\partial r^2}\bigl[\sigma^2 r\,p\bigr]$$

Solved numerically via **Crank–Nicolson finite differences** (second-order accurate, unconditionally stable). At each time step a tridiagonal system is solved in $O(N)$ operations via the Thomas algorithm.

![Fokker-Planck Density Evolution](fokker_planck_density.png)

*The density evolves from the initial short-rate spike toward the stationary Gamma distribution (black dashed).*

---

### 5. Edgeworth Series Approximation

For the non-central chi-squared transition density, a fourth-order Edgeworth expansion around a Gaussian baseline:

$$p_X(x) \approx \frac{\phi(z)}{\sigma_X}\!\left[1 + \frac{\kappa_3}{6\sigma_X^3}H_3(z) + \frac{\kappa_4}{24\sigma_X^4}H_4(z) + \frac{\kappa_3^2}{72\sigma_X^6}H_6(z)\right]$$

where $z = (x-\mu_X)/\sigma_X$, the Chebyshev–Hermite polynomials are:

$$H_3(z) = z^3-3z, \quad H_4(z) = z^4-6z^2+3, \quad H_6(z) = z^6-15z^4+45z^2-15$$

and cumulants $\kappa_3 = 8(q+1+3u)$, $\kappa_4 = 48(q+1+4u)$.

![Edgeworth vs Exact](edgeworth_vs_exact.png)

*The Edgeworth approximation (red dashed) is visually indistinguishable from the exact NcX2 density (blue) at t=0.5y.*

---

### 6. Bond Pricing: Feynman–Kac Formula

Zero-coupon bond price satisfies the PDE:

$$\frac{\partial P}{\partial t} + \kappa^*(\theta^*-r)\frac{\partial P}{\partial r} + \frac{\sigma^2 r}{2}\frac{\partial^2 P}{\partial r^2} - rP = 0, \qquad P(T,T)=1$$

Closed-form solution:

$$P(t,T) = A(\tau)^{2\kappa\theta/\sigma^2}\,e^{-B(\tau)r_t}$$

$$B(\tau) = \frac{2(e^{\gamma\tau}-1)}{(\gamma+\kappa^*)(e^{\gamma\tau}-1)+2\gamma}, \qquad A(\tau) = \frac{2\gamma\,e^{(\gamma+\kappa^*)\tau/2}}{(\gamma+\kappa^*)(e^{\gamma\tau}-1)+2\gamma}$$

$$\gamma = \sqrt{(\kappa^*)^2+2\sigma^2}, \quad \kappa^* = \kappa+\lambda\sigma, \quad \theta^* = \frac{\kappa\theta}{\kappa^*}$$

**Continuously-compounded yield** (the core prediction formula):

$$\boxed{y(\tau;\,r_t) = \underbrace{f(\tau)}_{\text{base level}} + \underbrace{g(\tau)}_{\text{sensitivity}}\cdot r_t}$$

where $$f(\tau) = -\log A(\tau)^{2\kappa\theta/\sigma^2}/\tau$ and $g(\tau) = B(\tau)/\tau$$.

---

### 7. Level, Slope, and Curvature Decomposition

Since the yield is affine in $r_t$, all three yield curve factors are deterministic functions of the single state variable:

$$\text{Level}(t) = \bar{f} + \bar{g}\,r_t \qquad (\text{average yield across maturities})$$

$$\text{Slope}(t) = [g(\tau_L)-g(\tau_S)]\,r_t + [f(\tau_L)-f(\tau_S)]$$

$$\text{Curvature}(t) = [2g(\tau_M)-g(\tau_S)-g(\tau_L)]\,r_t + [2f(\tau_M)-f(\tau_S)-f(\tau_L)]$$

**Critical result:** All three factors are perfectly correlated in the CIR model:

| | Level | Slope | Curvature |
|--|-------|-------|-----------|
| Level | 1.00 | −1.00 | 1.00 |
| Slope | −1.00 | 1.00 | −1.00 |
| Curvature | 1.00 | −1.00 | 1.00 |

This perfect correlation is both the model's defining structural property (tractability) and its binding limitation (cannot independently move slope and curvature).

Sensitivity ratio: $g(3M)/g(30Y) = 1.00/0.80 = 1.25\times$ — the short end moves 25% more than the long end per unit change in $r_t$.

![LSC Comparison](lsc_comparison.png)

*CIR Level, Slope, Curvature (blue) vs Nelson–Siegel empirical factors (orange). The CIR factors are smooth functions of $r_t$; the empirical factors exhibit rich multi-dimensional dynamics.*

![CIR Affine Decomposition](cir_affine.png)

*Left: CIR yield curves for different short rates. Right: affine decomposition $y(\tau;r) = f(\tau)+g(\tau)\cdot r$.*

---

## Bayesian Calibration: MCMC

### Log-Posterior

$$\log\pi(\kappa,\theta,\sigma \mid \mathbf{r}) = \underbrace{\sum_{t=1}^{T-1}\log p(r_{t+1}\mid r_t;\,\kappa,\theta,\sigma,\Delta t)}_{\text{exact NcX2 log-likelihood}} + \underbrace{\log\pi_0(\kappa,\theta,\sigma)}_{\text{log-prior}}$$

Priors: $\kappa\sim\text{Gamma}(2,1)$, $\theta\sim\text{Gamma}(2,0.05)$, $\sigma\sim\text{HalfNormal}(0.3)$

### Metropolis–Hastings

Log-space random-walk proposal (enforces positivity):
$$\log\psi^* = \log\psi^{(m)} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\delta^2 I_3)$$

Acceptance probability: $\alpha = \min\!\left(1,\,\pi(\psi^*\mid\mathbf{r})/\pi(\psi^{(m)}\mid\mathbf{r})\right)$

10,000 iterations, 3,000 burn-in, acceptance rate ~31%.

![MCMC Diagnostics](mcmc_diagnostics.png)

*Trace plots, posterior marginals, and chain ACF. $\sigma$ is well-identified; $\kappa$ and $\theta$ show slow mixing due to their posterior correlation.*

![Monte Carlo Paths](mc_paths.png)

*300 simulated one-year CIR paths. Red line = $\mathbb{E}[r_t\mid r_0]$. Dashed = $\hat\theta = 4.29\%$.*

---

## Parameter Estimation Robustness: OLS vs GMM vs MLE vs MCMC

### The Euler–Maruyama Bridge

$$\Delta r_t = \underbrace{\kappa\theta\Delta t}_{=\,a_0} - \underbrace{\kappa\Delta t}_{=\,\beta}\,r_t + \sigma\sqrt{r_t\Delta t}\,\varepsilon_{t+1}$$

This is an AR(1) with conditional variance $h_t = \sigma^2 r_t\Delta t$ — the bridge between the SDE and econometric models.

### OLS

Minimises $\sum_t(\Delta r_t - a_0 - a_1 r_t)^2$. Biased because $\text{Var}(\varepsilon_t) = \sigma^2 r_t\Delta t$ is heteroskedastic. OLS is only BLUE under homoskedasticity (Gauss–Markov), which is violated here.

### GMM (Two-Step Efficient)

Moment conditions from exact CIR moments:

$$g_1 = \mathbb{E}[r_{t+1}-\mathbb{E}[r_{t+1}\mid r_t]] = 0$$
$$g_2 = \mathbb{E}[(r_{t+1}-\mathbb{E}[r_{t+1}\mid r_t])^2 - \text{Var}(r_{t+1}\mid r_t)] = 0$$
$$g_3 = \mathbb{E}[r_t\cdot(r_{t+1}-\mathbb{E}[r_{t+1}\mid r_t])] = 0$$

Objective: $\min_\psi\;\mathbf{g}(\psi)'W\mathbf{g}(\psi)$, with $W = \hat{S}^{-1}$ in the second step.

### Comparison Table

| Method | $\hat\kappa$ | $\hat\theta$ (%) | $\hat\sigma$ | Feller | Log-Lik |
|--------|-------------|-----------------|-------------|--------|---------|
| OLS | −0.197 | −1.18 | 0.0412 | Yes | 13749.5 |
| GMM (2-step) | 0.030 | 20.00 | 0.0283 | Yes | — |
| MLE (Exact NcX2) | 0.000 | 857761 | 0.0423 | Yes | 13567.6 |
| MCMC (Bayesian) | 0.000 | 46.22 | 0.0419 | No | — |
| **Reference (Notebook 1)** | **0.047** | **4.29** | **0.0425** | **Yes** | — |

**Key insight:** Only $\sigma$ is reliably identified from daily data. Neither $\kappa$ nor $\theta$ individually is pinned down — only their product $\kappa\theta$ (which determines the drift magnitude) is well-identified. This is the near-unit-root identification problem for slow mean-reverting processes.

**Verdict:** MCMC with properly tuned proposals and sufficient iterations is preferred. It uses the exact likelihood (like MLE), provides full posterior uncertainty, and can be initialised with the MLE point estimate.

![Estimation Comparison](est_comparison.png)

*MCMC posterior (histogram), MLE (red), OLS (orange), GMM (green). $\sigma$ is well-identified by all methods; $\kappa$ and $\theta$ diverge dramatically.*

---

## Financial Econometrics

### AR(1)–GARCH(1,1)

The GARCH(1,1) generalises the CIR discretisation's heteroskedasticity:

$$h_t = \omega + \alpha_1\varepsilon_{t-1}^2 + \beta_1 h_{t-1}$$

MLE results: $\hat\alpha_1 = 0.200$, $\hat\beta_1 = 0.780$ (persistence = 0.98), Log-likelihood = 5704.46.

### EGARCH (log-linearised CIR variance equation)

$$\log h_t = \omega + \alpha_1(|\tilde\varepsilon_{t-1}|-\mathbb{E}[|\tilde\varepsilon|]) + \gamma_1\tilde\varepsilon_{t-1}\mathbf{1}_{\varepsilon<0} + \beta_1\log h_{t-1}$$

MLE results: $\hat\beta_1 = 0.971$, $\hat\alpha_1 = 0.762$, $\hat\gamma_1 = 0.005$ (insignificant, consistent with symmetric CIR diffusion). Log-likelihood = 6440.41 — better than GARCH.

![GARCH Conditional Volatility](garch_cond_vol.png)

*GARCH conditional volatility spikes in 2020 (COVID) and 2022–2023 (tightening cycle).*

### Johansen Cointegration Test

| H₀: rank ≤ r | Trace Stat | 5% CV | Reject? |
|---|---|---|---|
| r ≤ 0 | 530.52 | 197.38 | Yes |
| r ≤ 1 | 311.95 | 159.53 | Yes |
| r ≤ 2 | 206.85 | 125.62 | Yes |
| r ≤ 3 | 117.58 | 95.75 | Yes |
| r ≤ 4 | 68.58 | 69.82 | No |

Four cointegrating vectors found. CIR single-factor hypothesis predicts one; four indicates richer factor structure, motivating multi-factor extensions.

![Spreads and Correlations](spreads_corr.png)

*Left: yield spreads showing the 2022 inversion. Right: correlation matrix confirming high positive co-movement across all maturities.*

![Nelson–Siegel Factors](ns_factors.png)

*Level (top), slope (middle), curvature (bottom) — the three empirical dimensions of yield curve variation.*

---

## Advanced Extensions

### Two-Factor CIR (Longstaff–Schwartz 1992)

$$r_t = X_t + Y_t$$
$$dX_t = \kappa_1(\theta_1-X_t)\,dt + \sigma_1\sqrt{X_t}\,dW_t^{(1)}, \quad dY_t = \kappa_2(\theta_2-Y_t)\,dt + \sigma_2\sqrt{Y_t}\,dW_t^{(2)}$$

Bond price: $P = A_1(\tau)^{2\kappa_1\theta_1/\sigma_1^2}\,A_2(\tau)^{2\kappa_2\theta_2/\sigma_2^2}\,e^{-B_1(\tau)X_t-B_2(\tau)Y_t}$

Calibrated (MCMC):

| Factor | $\hat\kappa$ | $\hat\theta$ | $\hat\sigma$ | Half-life |
|--------|-------------|-------------|-------------|----------|
| $X_t$ (level) | 0.108 | 5.10% | 0.054 | 6.4 years |
| $Y_t$ (cycle) | 0.416 | 0.96% | 0.100 | 1.7 years |

### Jump-Diffusion CIR (Duffie–Pan–Singleton 2000)

$$dr_t = \kappa(\theta-r_t)\,dt + \sigma\sqrt{r_t}\,dW_t + J_t\,dN_t$$

where $N_t\sim\text{Poisson}(\lambda_J)$, $J_t\sim\text{Exp}(\mu_J)$. Bond price remains affine with Riccati ODE:

$$\frac{d\alpha}{d\tau} = \kappa\theta\,\beta + \lambda_J\!\left(\frac{1}{1-\mu_J\beta}-1\right), \qquad \frac{d\beta}{d\tau} = 1-\kappa\beta-\frac{\sigma^2}{2}\beta^2$$

Calibration result: $\hat\lambda_J \approx 0$ — the jump model collapses to the base CIR at daily frequency. Jump parameters require options data or a much longer sample with multiple large-shock episodes.

### CIR++ (Brigo–Mercurio 2001)

$$y^{++}(\tau;\,r_t) = y^\text{CIR}(\tau;\,r_t) + \hat\phi(\tau), \qquad \hat\phi(\tau) = \frac{1}{T}\sum_t[y_\tau^\text{actual}-y_\tau^\text{CIR}]$$

| Maturity | Shift $\hat\phi$ (bps) |
|----------|----------------------|
| 6M | +13.6 |
| 9M | +20.0 |
| 1Y | +26.5 |
| 2Y | +15.5 |
| 5Y | +17.0 |
| 10Y | +41.7 |
| 20Y | +79.9 |
| 30Y | +94.0 |

All shifts positive: the base CIR systematically underpredicts all maturities in training, reflecting the nearly flat yield curve implied by $\kappa\approx 0$.

---

## Statistical Learning Models

### Universal Approximation by Trees (Stone 1977)

For any Lipschitz function $f:[\tau_\min,\tau_\max]\times[r_\min,r_\max]\to\mathbb{R}$ and $\varepsilon>0$, there exists a regression tree $g$ such that $\sup|f-g|<\varepsilon$.

The CIR yield is Lipschitz in $(\tau,r)$, so trees can approximate it arbitrarily well. **However**, the approximation is only valid within the training support.

### Gradient Boosting

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\,h_m(\mathbf{x}), \quad h_m = \arg\min_h\sum_i[\tilde r_i^{(m)}-h(\mathbf{x}_i)]^2$$

Pseudo-residuals: $\tilde r_i^{(m)} = y_i - F_{m-1}(\mathbf{x}_i)$ (for MSE loss).

### Random Forest Bias-Variance Decomposition

$$\mathbb{E}[(\hat f_{RF}-f)^2] = \text{Bias}^2 + \rho\sigma_T^2 + \frac{(1-\rho)}{B}\sigma_T^2 \;\xrightarrow{B\to\infty}\; \text{Bias}^2+\rho\sigma_T^2$$

Feature importance for 10Y prediction: `theory_y`, `r_B`, `r3m` dominate with ~99% importance — the ML model essentially learns a correction to the CIR prediction.

![Feature Importance](feature_importance.png)

*CIR-derived features dominate ML feature importance, confirming the CIR structure is the key signal.*

---

## Prediction Results

**Protocol:** Only the 3M yield is used as input. The model reconstructs yields at all other maturities.

### Prophet-Style Prediction Plots

![All Maturities](prophet_all.png)

*Prophet-style predictions for all test maturities. Shaded bands = 68% and 95% Bayesian credible intervals from MCMC posterior.*

![3M Detail](prophet_3m.png)

*Detailed 3M prediction (trivially exact since it is the input). The two-factor model shows minor deviation from the input due to factor decomposition.*

### Out-of-Sample Performance

| Model | 3M R² | 6M R² | 9M R² | 1Y R² | 2Y R² | Mean R² | Mean RMSE (bp) |
|-------|--------|--------|--------|--------|--------|---------|----------------|
| **Base CIR (MCMC)** | **1.000** | **0.986** | **0.906** | **0.668** | **−0.844** | **0.543** | **23.5** |
| CIR++ | 1.000 | 0.938 | 0.738 | 0.220 | −1.368 | 0.305 | 33.4 |
| Two-Factor CIR | 0.993 | 0.944 | 0.768 | 0.320 | −2.831 | 0.039 | 36.8 |
| Jump-Diffusion CIR | 1.000 | 0.986 | 0.906 | 0.668 | −0.844 | 0.543 | 23.5 |
| Spline (ML) | — | — | — | — | — | −0.044 | 55.2 |
| Gradient Boost | — | — | — | — | — | −0.431 | 64.5 |
| Random Forest | — | — | — | — | — | −0.433 | 64.1 |

The R² target of 0.85 is achieved at 6M (R² = 0.986) and 9M (R² = 0.906) by the base CIR model. No ML model meets the target at any maturity.

![Extensions Comparison](ext_comparison.png)

*R² and RMSE by maturity for all CIR extensions. Base CIR dominates. Target R² = 0.85 (dashed).*

![Model Comparison](model_comparison.png)

*Full model comparison including ML methods. The CIR model's structural constraints beat all data-driven alternatives.*

### Yield Curve Prediction Detail

![Yield Curve Prediction](yield_curve_pred.png)

*Predicted vs actual yield curves on the final test date. CIR (blue) tracks the actual curve (black) most closely among all methods.*

### Historical EDA

![EDA Yields](eda_yields.png)

*Historical yield time series for all nine maturities (training data 2016–2024).*

![Yield Surface](yield_surface.png)

*3D yield surface revealing the time-varying level, slope, and curvature of the term structure.*

![ACF PACF](acf_pacf.png)

*ACF/PACF of first-differenced yields. Short end shows mild serial dependence; long end is nearly white noise.*

---

## Why CIR Beats ML

The CIR model outperforms all ML alternatives because:

1. **Economic constraints act as regularisation.** Mean reversion toward $\theta$, positivity under the Feller condition, and the affine yield structure are all correct in this data regime. These constraints prevent economically nonsensical extrapolations.

2. **ML models cannot extrapolate.** Tree-based models are bounded by the training data range. The test period features rate levels near the maximum of training, causing degradation precisely where the ML models need to extrapolate.

3. **The dominant factor is well-captured.** The test period is dominated by level shifts in rates. A single-factor model that captures the level well will outperform models adding slope/curvature flexibility at the cost of level estimation accuracy.

4. **CIR features dominate ML importance.** Feature importance confirms that even the ML models effectively learn the CIR structure and add non-generalising corrections on top.

---

## Limitations and Risk Management Implications

### Theoretical Limitations

| Limitation | Consequence | Extension |
|---|---|---|
| Single factor | Only parallel shifts; slope/curvature move perfectly correlated with level | Two-factor CIR (Longstaff–Schwartz 1992) |
| No jumps | Misses central bank announcement effects | Jump-diffusion (Duffie–Pan–Singleton 2000) |
| Constant parameters | Cannot fit arbitrary initial term structure | CIR++ (Brigo–Mercurio 2001) |
| Zero lower bound | Near-zero rates approach degenerate diffusion | Shifted CIR, Hull–White |
| Risk-neutral calibration | Physical and risk-neutral measures give different $(\kappa,\theta)$ | Market price of risk $\lambda$ identification |

### Analytical Risk Metrics (Closed Form)

Rate sensitivity (DV01):
$$\text{DV01}(\tau) = B(\tau)\,P(t,T) \times 10^{-4}$$

Convexity:
$$\Gamma(\tau) = B(\tau)^2\,P(t,T)$$

These closed-form expressions are among the CIR model's most valuable practical contributions for hedging and risk measurement.

### Trading Implications

The CIR model is appropriate for level risk hedging, bond pricing, and short-horizon Monte Carlo VaR. It is not appropriate for slope/curvature risk (needs multi-factor), negative rate environments (needs shifted or Hull–White), or options pricing requiring the full volatility smile (needs SABR or HJM).

---

## Key Equations Quick Reference

| Quantity | Formula |
|----------|---------|
| CIR SDE | $dr_t = \kappa(\theta-r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$ |
| Feller condition | $2\kappa\theta \geq \sigma^2$ |
| Conditional mean | $\mathbb{E}[r_t\mid r_0] = \theta + (r_0-\theta)e^{-\kappa t}$ |
| Half-life | $t_{1/2} = \log 2/\kappa$ |
| Bond price | $P(t,T) = A(\tau)^{2\kappa\theta/\sigma^2}e^{-B(\tau)r_t}$ |
| Yield (prediction formula) | $y(\tau;r_t) = [B(\tau)r_t - \log A(\tau)^{2\kappa\theta/\sigma^2}]/\tau$ |
| Stationary distribution | $r_\infty \sim \text{Gamma}(2\kappa\theta/\sigma^2,\;\sigma^2/(2\kappa))$ |
| GARCH(1,1) | $h_t = \omega + \alpha_1\varepsilon_{t-1}^2 + \beta_1 h_{t-1}$ |
| 2-factor yield | $y(\tau;X,Y) = [B_1 X + B_2 Y - \log A_1^{2k_1\theta_1/\sigma_1^2} - \log A_2^{2k_2\theta_2/\sigma_2^2}]/\tau$ |
| Jump-diffusion Riccati | $d\beta/d\tau = 1 - \kappa\beta - \sigma^2\beta^2/2$ |
| CIR++ correction | $y^{++} = y^\text{CIR} + \hat\phi(\tau)$ |

---

## References

- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). A theory of the term structure of interest rates. *Econometrica*, 53(2), 385–407.
- Brigo, D., Mercurio, F. (2001). *Interest Rate Models: Theory and Practice*. Springer Finance.
- Longstaff, F.A., Schwartz, E.S. (1992). Interest rate volatility and the term structure. *Journal of Finance*, 47(4), 1259–1282.
- Duffie, D., Pan, J., Singleton, K. (2000). Transform analysis and asset pricing for affine jump-diffusions. *Econometrica*, 68(6), 1343–1376.
- Nelson, C.R., Siegel, A.F. (1987). Parsimonious modelling of yield curves. *Journal of Business*, 60(4), 473–489.
- Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Stone, C.J. (1977). Consistent nonparametric regression. *Annals of Statistics*, 5(4), 595–645.
