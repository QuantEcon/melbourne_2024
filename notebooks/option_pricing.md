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

+++

In this notebook we use option pricing as an application to test some Python scientific computing libraries.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Introduction to Monte Carlo integration

Before discussing option pricing we'll quickly review Monte Carlo integration and why it's useful.

### Computing expectations

Suppose that we want to evaluate

$$
    \mathbb E f(X)
$$

where $X$ is a random variable or vector and $f$ is some given function.

This is easy in some cases

For example, if $f(x) = x^2$ and $X \sim N(0,1)$, then 

$$
    \mathbb E f(X) 
    = \mathbb E X^2
    = \mathbb E (X - \mathbb E X)^2
    = 1
$$

But what if $f$ is a more complex function, such as

$$f(x) = \ln(1 + X + \exp(X) + X^{1/2})$$ 

Let's also suppose that $X$ is exponential, meaning that its density is

$$
g(x) = \lambda \exp(-\lambda x)
$$

How would you compute $\mathbb E f(X)$ in this case?

+++

### Numerical integration

One option is to use numerical integration.

We want to integrate $h(x) = f(x) g(x)$ where $g$ is the exponential density.

That is, we want to compute

$$
\mathbb E f(X) 
    = \int_0^\infty f(x) g(x) d x
    = \int_0^\infty h(x) d x
$$

First we define $h$

```{code-cell} ipython3
def g(x, λ=1.0):
    return λ * np.exp(- λ * x)

def f(x):
    return np.log(1 + x + np.exp(x) + np.sqrt(x)) 

def h(x):
    return f(x) * g(x)
```

Let's plot $h$ to see what it looks like.

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
a, b, n = 0, 5, 6
integral, x, y = trapezoidal_rule(h, a=a, b=b, n=n)
```

```{code-cell} ipython3
integral
```

```{code-cell} ipython3
# Plot function
x_full = np.linspace(a, b, 1000)
y_full = h(x_full)
fig, ax = plt.subplots()
ax.plot(x_full, y_full, 'r', linewidth=2, label='$h(x)$')

# Plot trapezoids
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

What if I tell you that $X$ is created as follows:

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

### Set up

The price satisfies

$$
P = \beta^n \mathbb E \max\{ S_n - K, 0 \}
$$

where

1. $\beta$ is a discount factor,
2. $n$ is the expiry date,
2. $K$ is the strike price and
3. $S_n$ is the price of the underlying asset after $n$ periods.

For example, consider a call option to buy stock in Amazon at strike price $K$. 

The owner has the right (but not the obligation) to buy 1 share in Amazon at
price $K$ after $n$ days.  

The payoff is therefore $\max\{S_n - K, 0\}$

The risk-neutral price is the expectation of the payoff, discounted to current value.

Notice that this is another example of computing $P = \mathbb E f(X)$

In all of what follows we will use

```{code-cell} ipython3
n, β, K = 10, 0.99, 100
```

It remains only to specify the distribution of $S_n$.

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

We use the default values

```{code-cell} ipython3
μ, ρ, ν, S_0, h_0 = 0.0001, 0.1, 0.001, 10.0, 0.0
```

Let's plot 12 of these paths:


```{code-cell} ipython3
M, n = 12, 10
fig, axes = plt.subplots(2, 1, figsize=(6, 8))
s_0 = np.log(S_0)
for m in range(M):
    s = np.empty(n+1)
    s[0], h = s_0, h_0
    for t in range(n):
        s[t+1] = s[t] + μ + np.exp(h) * np.random.randn()
        h = ρ * h + ν * np.random.randn()
    axes[0].plot(s)
    axes[1].plot(np.exp(s))
axes[0].set_title('log share price over time')
axes[1].set_title('share price over time')
plt.show()
```

Here's a larger simulation, where we 

* set $M = 1000$ and
* generate $M$ draws of $s_n$

```{code-cell} ipython3
M, n = 1_000, 10
s_0 = np.log(S_0)
s_n = np.empty(M)
for m in range(M):
    s, h = s_0, h_0
    for t in range(n):
        s = s + μ + np.exp(h) * np.random.randn()
        h = ρ * h + ν * np.random.randn()
    s_n[m] = s
```

Let's histogram the $M$ values of $s_n$

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.hist(s_n, bins=25, alpha=0.5)
plt.show()
```

Actually what we want is $S_n = \exp(s_n)$, so let's look at the distribution.

```{code-cell} ipython3
S_n = np.exp(s_n)
fig, ax = plt.subplots()
ax.hist(S_n, bins=25, alpha=0.5)
plt.show()
```

We can see that it's heavy-tailed

* many small observations
* a few very large ones


### Computing the price of the option

Now we have observations of the share price, we can get an estimate of the option price via

$$
    \hat P_M 
    := \beta^n \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \beta^n \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
$$

```{code-cell} ipython3
price = β**n * np.mean(np.maximum(S_n - K, 0))
price 
```

Let's write a function to do this

We'll use the following default for $M$

```{code-cell} ipython3
medium_M = 1_000_000
```

```{code-cell} ipython3
def compute_call_price_py(β=β,
                           μ=μ,
                           S_0=S_0,
                           h_0=h_0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=medium_M,
                           seed=1234):
    np.random.seed(seed)
    s_0 = np.log(S_0)
    s_n = np.empty(M)
    for m in range(M):
        s, h = s_0, h_0
        for t in range(n):
            s = s + μ + np.exp(h) * np.random.randn()
            h = ρ * h + ν * np.random.randn()
        s_n[m] = s
    S_n = np.exp(s_n)
    expectation = np.mean(np.maximum(S_n - K, 0))

    return β**n * expectation
```

Let's try computing the price

```{code-cell} ipython3
%time compute_call_price_py(seed=1)
```

The runtime is very long, even with moderate sample size $M$

Moreover, the sample size is still too small!

To see this, let's try again with a different seed

```{code-cell} ipython3
%time compute_call_price_py(seed=2)
```

Notice the big variation in the price --- the variance of our estimate is too high.

+++

## NumPy Version

```{code-cell} ipython3
def compute_call_price_np(β=β,
                          μ=μ,
                          S_0=S_0,
                          h_0=h_0,
                          K=K,
                          n=n,
                          ρ=ρ,
                          ν=ν,
                          M=medium_M,
                          seed=1234):
    np.random.seed(seed)
    s = np.full(M, np.log(S_0))
    h = np.full(M, h_0)
    for t in range(n):
        Z = np.random.randn(2, M)
        s = s + μ + np.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = np.mean(np.maximum(np.exp(s) - K, 0))
        
    return β**n * expectation
```

Now computation of the option price estimate is much faster.

```{code-cell} ipython3
%time compute_call_price_np(seed=1)
```

This means that we can estimate the price using a serious sample size

```{code-cell} ipython3
large_M = 10 * medium_M
%time compute_call_price_np(M=large_M, seed=1)
```

Let's try with a different seed to get a sense of the variance.

```{code-cell} ipython3
%time compute_call_price_np(M=large_M, seed=2)
```

OK, the sample size is *still* too small, which tells us that we need more
speed.

Let's leave $M$ fixed at `large_M` but try to make the routine faster

+++

## Parallel Numba version

Let's try a Numba version with parallelization.

```{code-cell} ipython3

import numba
from numba import prange

@numba.jit(parallel=True)
def compute_call_price_numba(β=β,
                               μ=μ,
                               S_0=S_0,
                               h_0=h_0,
                               K=K,
                               n=n,
                               ρ=ρ,
                               ν=ν,
                               M=medium_M,
                               seed=1234):
    np.random.seed(seed)
    s_0 = np.log(S_0)
    s_n = np.empty(M)
    for m in prange(M):
        s, h = s_0, h_0
        for t in range(n):
            s = s + μ + np.exp(h) * np.random.randn()
            h = ρ * h + ν * np.random.randn()
        s_n[m] = s
    S_n = np.exp(s_n)
    expectation = np.mean(np.maximum(S_n - K, 0))

    return β**n * expectation
```

```{code-cell} ipython3
%time compute_call_price_numba(M=large_M)
```

```{code-cell} ipython3
%time compute_call_price_numba(M=large_M)
```

## JAX Version

```{code-cell} ipython3
!nvidia-smi
```

The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

### Simple JAX version

Let's start with a simple version that looks like the NumPy version.

```{code-cell} ipython3
def compute_call_price_jax(β=β,
                           μ=μ,
                           S_0=S_0,
                           h_0=h_0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=1_000_000,
                           seed=1234):

    key=jax.random.PRNGKey(seed)
    s = jnp.full(M, np.log(S_0))
    h = jnp.full(M, h_0)
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
price = compute_call_price_jax(M=large_M)
print(price)
```

And now let's time it:

```{code-cell} ipython3
%%time 
price = compute_call_price_jax(M=large_M)
print(price)
```

### Compiled JAX version

+++

Let's take the simple version above and compile the entire function using JAX

```{code-cell} ipython3
compute_call_price_jax_compiled = jax.jit(compute_call_price_jax, static_argnums=(8, ))
```

```{code-cell} ipython3
%%time 
price = compute_call_price_jax_compiled(M=large_M)
print(price)
```

```{code-cell} ipython3
%%time 
price = compute_call_price_jax_compiled(M=large_M)
print(price)
```

