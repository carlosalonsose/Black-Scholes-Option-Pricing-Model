# Black–Scholes Option Pricing Model

This project implements the **Black–Scholes model** for European options.  
It calculates **call & put prices**, the **Greeks**, solves for **implied volatility** using a gradient-based method and demostrate the Call-Put parity.  
The notebook also includes **explanations, derivations, and visualizations** — making it both a **quant finance reference** and a **learning resource**.

---

## Features

- **Closed-form Black–Scholes prices** (calls & puts)  
- **Greeks**:  
  - $\boldsymbol{\Delta}$ (Delta)  
  - $\boldsymbol{\Gamma}$ (Gamma)  
  - $\boldsymbol{\nu}$ (Vega)  
  - $\boldsymbol{\Theta}$ (Theta)  
  - $\boldsymbol{\rho}$ (Rho)  
Computed with `jax.grad`  
- **Implied Volatility** solved numerically (**Newton–Raphson method**)  
- **Plots** showing how price & sensitivities move with **volatility**, **maturity**, and **strike**  
- **Clear mathematical explanations** and **arbitrage logic** (put–call parity)

---

## Quick Overview

The **Black–Scholes model** takes six inputs:

- **$\boldsymbol{S}$**: Current underlying price  
- **$\boldsymbol{K}$**: Strike price  
- **$\boldsymbol{T}$**: Time to maturity (years)  
- **$\boldsymbol{r}$**: Risk-free rate  
- **$\boldsymbol{σ}$**: Volatility  
- **$\boldsymbol{q}$**: Dividend yield  

**Formulas (Call & Put):**

$$
\mathbf{C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)}
$$

$$
\mathbf{P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)}
$$

where $\boldsymbol{d_1}$ and $\boldsymbol{d_2}$ measure option *moneyness* adjusted for volatility and time.

---

## In-Depth Explanation

For a full walkthrough of:
- Black–Scholes $\boldsymbol{PDE}$
- Derivation of $\boldsymbol{d_1}$ and $\boldsymbol{d_2}$
- Risk-neutral interpretation of probabilities
- Numerical search for implied volatility
- Arbitrage logic with **put–call parity**

See the detailed sections inside the notebook.

---

## Technologies
- Python 3
- NumPy / SciPy
- JAX (for automatic differentiation)
- Matplotlib

---

## Project Structure
- `Black-Scholes.ipynb` → main notebook with code, math, and plots

---

## Applications
- Option pricing in derivatives markets
- Sensitivity/risk analysis for traders (Greeks)
- Inferring volatility from market prices (IV)
- Teaching/learning quantitative finance
