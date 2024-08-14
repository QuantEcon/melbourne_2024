---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Pricing a European Call Option Version 2

----

#### John Stachurski (August 2024)

----

+++

In this notebook we will accelerate our code for option pricing using different
libraries

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Why is Pure Python Slow?

We saw that our Python code for option pricing was pretty slow.

In essence, this is because our loops were written in pure Python

Pure Python loops are not fast.

This has led some people to claim that Python is too slow for computational
economics.

These people are ~idiots~ misinformed -- so please ignore them.

Evidence: AI teams are solving optimization problems in $\mathbb R^d$ with $d >
$ 1 trillion using Pytorch / JAX

So I'm pretty sure we can use Python for computational economics.

But first let's try to understand the issues.

### Issue 1: Type Checking

Consider the following Python code

```{code-cell} ipython3
x, y = 1, 2
x + y
```

This is integer addition, which is different from floating point addition

```{code-cell} ipython3
x, y = 1.0, 2.0
x + y
```

Now consider this code

```{code-cell} ipython3
x, y = 'foo', 'bar'
x + y
```

Notice that we use the same symbol `+` on each occasion.

The Python interpreter figures out the correct action by type checking:

```{code-cell} ipython3
a, b = 'foo', 10
type(a)
```

```{code-cell} ipython3
type(b)
```

But think of all the type checking in our option pricing function --- the
overhead is huge!!

```{code-cell} ipython3
n, β, K = 10, 0.99, 100
μ, ρ, ν, S_0, h_0 = 0.0001, 0.01, 0.001, 10.0, 0.0
def compute_call_price_py(β=β,
                           μ=μ,
                           S_0=S_0,
                           h_0=h_0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=1_000_000,
                           seed=1234):
    np.random.seed(seed)

    s_0 = np.log(S_0)
    s_n = np.empty(M)

    for m in range(M):
        s, h = s_0, h_0
        for t in range(n):
            U, V = np.random.randn(2)
            s = s + μ + np.exp(h) * U
            h = ρ * h + ν * V
        s_n[m] = s

    S_n = np.exp(s_n)

    expectation = np.mean(np.maximum(S_n - K, 0))

    return β**n * expectation
```

### Issue 2:  Memory Management

Pure Python emphasizes flexibility and hence cannot attain maximal efficiency
vis-a-vis memory management.

For example,

```{code-cell} ipython3
import sys
x = [1.0, 2.0]  
sys.getsizeof(x) * 8   # number of bits
```

### Issue 3:  Parallelization

There are opportunities to parallelize our code above -- divide it across
multiple workers.

This can't be done efficiently with pure Python but certainly can with the right
Python libraries.

+++

## Vectorization

As a first pass at improving efficiency, here's a vectorized version where all paths are updated together.

We use NumPy to store and update each vector of share prices.

When we use NumPy, type-checking is done per-array, not per element!

```{code-cell} ipython3
def compute_call_price_np(β=β,
                          μ=μ,
                          S_0=S_0,
                          h_0=h_0,
                          K=K,
                          n=n,
                          ρ=ρ,
                          ν=ν,
                          M=10_000_000,
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

Now computation of the option price is reasonably fast:

```{code-cell} ipython3
%time compute_call_price_np()
```

But we can still do better...

+++

## Numba Version

Let's try a Numba version.

This version uses a just-in-time (JIT) compiler to eliminate type-checking.

```{code-cell} ipython3
import numba

@numba.jit()
def compute_call_price_numba(β=β,
                               μ=μ,
                               S_0=S_0,
                               h_0=h_0,
                               K=K,
                               n=n,
                               ρ=ρ,
                               ν=ν,
                               M=10_000_000,
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

```{code-cell} ipython3
%time compute_call_price_numba()
```

```{code-cell} ipython3
%time compute_call_price_numba()
```

## Numba Plus Parallelization

The last version was only running on one core.

Next let's try a Numba version with parallelization.

```{code-cell} ipython3
from numba import prange

@numba.jit(parallel=True)
def compute_call_price_numba_parallel(β=β,
                                      μ=μ,
                                      S_0=S_0,
                                      h_0=h_0,
                                      K=K,
                                      n=n,
                                      ρ=ρ,
                                      ν=ν,
                                      M=10_000_000,
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
%time compute_call_price_numba_parallel()
```

```{code-cell} ipython3
%time compute_call_price_numba_parallel()
```

## JAX Version

We can do even better if we exploit a hardware accelerator such as a GPU.

Let's see if we have a GPU:

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
                           M=10_000_000,
                           seed=1234):

    key=jax.random.PRNGKey(seed)
    s = jnp.full(M, np.log(S_0))
    h = jnp.full(M, h_0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    S = jnp.exp(s)
    expectation = jnp.mean(jnp.maximum(S - K, 0))
        
    return β**n * expectation
```

Let's run it once to compile it:

```{code-cell} ipython3
%%time 
price = compute_call_price_jax()
print(price)
```

And now let's time it:

```{code-cell} ipython3
%%time 
price = compute_call_price_jax()
print(price)
```

### Compiled JAX version

+++

Let's take the simple JAX version above and compile the entire function.

```{code-cell} ipython3
compute_call_price_jax_compiled = jax.jit(compute_call_price_jax, static_argnums=(8, ))
```

We run once to compile.

```{code-cell} ipython3
%%time 
price = compute_call_price_jax_compiled()
print(price)
```

And now let's time it.

```{code-cell} ipython3
%%time 
price = compute_call_price_jax_compiled()
print(price)
```

Now we have a really big speed gain relative to NumPy.
