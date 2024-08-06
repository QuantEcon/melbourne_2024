---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Coding Application: Pricing a European Call Option

----

#### John Stachurski (August 2024)

----

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Introduction to Monte Carlo integration

### Computing expectations

Suppose that we want to evaluate

$$
P = \mathbb E f(X)
$$

where $X$ is a random variable or vector and $f$ is some given function.

This is easy in some cases

* e.g., $f(x) = x^2$ and $X \sim N(0,1)$ $\implies$ $\mathbb E f(X) = 1$.

But what if 

$$f(x) = \ln(1 + X + \exp(X) + X^{1/2})$$ 

and $X$ is exponential?

How would you compute $\mathbb E f(X)$ in this case?

### Numerical integration

One option here is to use a numerical integration method 

+++

We want to integrate $h(x) = f(x) g(x)$ where $g$ is the exponential density.

```{code-cell} ipython3
def g(x, λ=1.0):
    return λ * np.exp(- λ * x)

def f(x):
    return np.log(1 + x + np.exp(x) + np.sqrt(x)) * g(x)

def h(x):
    return f(x) * g(x)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
a, b = 0, 8
x_full = np.linspace(a, b, 1000)
y_full = h(x_full)
ax.plot(x_full, y_full, 'r', linewidth=2, label='$h(x)$')
ax.legend()
plt.show()
```

Here's a really simple numerical integration routine that uses the trapezoidal rule.

```{code-cell} ipython3
def trapezoidal_rule(u, a=0, b=5, n=6):
    """
    Approximate the integral of u over [a, b] using the trapezoidal rule.
    """
    x = np.linspace(a, b, n+1)
    y = u(x)
    h = (b - a) / n
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral, x, y
```

Let's integrate:

```{code-cell} ipython3
integral, x, y = trapezoidal_rule(h)
```

```{code-cell} ipython3
integral
```

```{code-cell} ipython3
# Plot function
a, b = x.min(), x.max()
x_full = np.linspace(a, b, 1000)
y_full = h(x_full)
fig, ax = plt.subplots()
ax.plot(x_full, y_full, 'r', linewidth=2, label='$h(x)$')

# Plot trapezoids
n = len(x) + 1
for i in range(n):
    x0 = a + i * (b - a) / n
    x1 = a + (i + 1) * (b - a) / n
    ax.fill_between([x0, x1], [0, 0], [h(x0), h(x1)], 
                    color='blue', alpha=0.3, edgecolor='black')

ax.set_title(f'estimated integral with {n} grid points = {integral:.3f}')
ax.set_xlabel('$x$')
ax.set_ylabel('$h(x)$')
ax.legend()
plt.show()
```

OK, so we figured out how to handle the problem above numerically.

But now let's make it harder.

What if I tell you that X is created as follows:

1. $\sigma$ is drawn from the exponential distribution with rate $\lambda = 2.0$
2. $\mu$ is drawn from a Beta$(a, b)$ distribution where $a=1.0$ and $b=3.0$
3. $Z$ is drawn as $\exp(Y)$ where $Y$ is $N(\mu, \sigma)$
4. $X$ is taken as the minimum of $Z$ and $2.0$

Now how would you compute $\mathbb E f(X)$?

```{code-cell} ipython3
for i in range(20):
    print("Solution below!")
```

**Solution**

To solve the problem numerically we can use Monte Carlo:

1. Generate $n$ IID draws $(X_i)$ of $X$
2. Approximate $\mathbb E f(X)$ via $(1/n) \sum_{i=1}^n f(X_i)$

```{code-cell} ipython3
def draw_x():
    σ = np.random.exponential(scale=1/2)
    μ = np.random.beta(a=1.0, b=3.0)
    Y = μ + σ * np.random.randn()
    return np.minimum(np.exp(Y), 2.0)
```

```{code-cell} ipython3
x_samples = [draw_x() for i in range(10_000)]
x_samples = np.array(x_samples)
```

```{code-cell} ipython3
np.mean(f(x_samples))
```

Of course, if we want a better approximation, we should generate more samples.

+++

## Pricing a call option

Now we're ready to price a European call option under the assumption of risk neutrality.

The price satisfies

$$
P = \beta^n \mathbb E \max\{ S_n - K, 0 \}
$$

where

1. $\beta$ is a discount factor,
2. $n$ is the expiry date,
2. $K$ is the strike price and
3. $\{S_t\}$ is the price of the underlying asset at each time $t$.

For example, consider a call option to buy stock in Amazon at strike price $K$. 

The owner has the right (but not the obligation) to buy 1 share in Amazon at price $K$ after $n$ days.  

The payoff is therefore $\max\{S_n - K, 0\}$

The risk-neutral price is the expectation of the payoff, discounted to current value.

Notice that this is another example of computing $P = \mathbb E f(X)$


+++

### Distribution of the share price

Often the distribution of $S_n$ is not a simple distribution.

As one example, let's set $s_t = \ln S_t$ and assume that the log stock price obeys 

$$ 
\ln s_{t+1} = \ln s_t + \mu + \sigma_t \xi_{t+1}
$$

where 

$$ 
    \sigma_t = \exp(h_t), 
    \quad
        h_{t+1} = \rho h_t + \nu \eta_{t+1}
$$

Here $\{\xi_t\}$ and $\{\eta_t\}$ are IID and standard normal.

(This is a **stochastic volatility** model, where the volatility $\sigma_t$
varies over time.)

Here's a simulation of the paths:


+++

Use the defaults `μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0`.

(Here `S0` is $S_0$ and `h0` is $h_0$.)

By generating $M$ paths $s_0, \ldots, s_n$, compute the Monte Carlo estimate 

$$
    \hat P_M 
    := \beta^n \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$
    

+++

## NumPy Version

+++

## NumPy + Numba Version


of the price, applying Numba and parallelization.

+++

## JAX Version

```{code-cell} ipython3
!nvidia-smi
```

The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

```{code-cell} ipython3
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.PRNGKey(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))
        
    return β**n * expectation
```

Let's run it once to compile it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

And now let's time it:

```{code-cell} ipython3
%%time 
compute_call_price_jax().block_until_ready()
```

```{solution-end}
```
