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

# Job Search

----
#### John Stachurski (August 2024)

----

```{code-cell} ipython3
#!pip install quantecon
```

In this lecture we study a basic infinite-horizon job search problem with Markov wage
draws 

The exercise at the end asks you to add recursive preferences and compare
the result.

We use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import quantecon as qe
import jax
import jax.numpy as jnp
from collections import namedtuple

jax.config.update("jax_enable_x64", True)
```

Let's check our GPU status:

```{code-cell} ipython3
!nvidia-smi
```

## Model

We study an elementary model where 

* jobs are permanent 
* unemployed workers receive current compensation $c$
* the wage offer distribution $\{W_t\}$ is Markovian
* the horizon is infinite
* an unemployment agent discounts the future via discount factor $\beta \in (0,1)$


An unemployed worker tries to maximize an expected sum of discounted lifetime payoffs.

+++

### Set up

We consider a wage offer process

$$
    W_{t+1} = \rho W_t + \nu Z_{t+1}
$$

where $(Z_t)_{t \geq 0}$ is IID and standard normal.

We discretize this wage process using Tauchen's method to produce

* an $n \times n$ stochastic matrix $P$ and
* a set of possible wage values $\{w_1, \ldots, w_n\}$

Since jobs are permanent, the return to accepting wage offer $w$ today is

$$
    w + \beta w + \beta^2 w + \cdots = \frac{w}{1-\beta}
$$

The worker chooses between accepting and rejecting in order to maximize expected lifetime value.

+++

### The Bellman equation

The Bellman equation is

$$
    v(w) = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

The solution to this equation is called the **value function** and we denote it $v^*$.

It is known that a policy is optimal if and only if 

$$
    \sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v^*(w') P(w, w')
        \right\}
$$

Here $\mathbf 1$ is an indicator function.

* $\sigma(w) = 1$ means stop (accept offer)
* $\sigma(w) = 0$ means continue (reject).

+++

### Algorithm

We solve this model using value function iteration.

This means that we use the Bellman operator

$$
    (Tv)(w) = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

The steps are

1. pick an initial guess $v$
2. iterate with $T$ to produce $v_k = T^k v$
3. choose a $v_k$ **greedy** policy $\sigma$, meaning that $\sigma$ satisfies

$$
    \sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v_k(w') P(w, w')
        \right\}
$$

+++

## Code

Let's set up a namedtuple to store information needed to solve the model.

```{code-cell} ipython3
Model = namedtuple('Model', 
                   ('n',        # wage grid size
                    'w_vals',   # wage values 
                    'P',        # transition matrix
                    'β',        # discount factor
                    'c'))       # unemployment compensation
```

The function below holds default values and populates the namedtuple.

```{code-cell} ipython3
def create_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), jnp.array(mc.P)
    return Model(n, w_vals, P, β, c)
```

Let's test it:

```{code-cell} ipython3
model = create_js_model(β=0.98)
```

```{code-cell} ipython3
model.c
```

```{code-cell} ipython3
model.β
```

```{code-cell} ipython3
model.w_vals.mean()  
```

Here's the Bellman operator.

```{code-cell} ipython3
@jax.jit
def T(v, model):
    """
    The Bellman operator Tv = max{e, c + β E v} with 

        e(w) = w / (1-β) and (Ev)(w) = E_w[ v(W')]

    """
    n, w_vals, P, β, c = model
    h = c + β * P @ v
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)
```

The next function computes the optimal policy under the assumption that $v$ is
                 the value function.

The policy takes the form

$$
    \sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v(w') P(w, w')
        \right\}
$$

```{code-cell} ipython3
@jax.jit
def get_greedy(v, model):
    "Get a v-greedy policy."
    n, w_vals, P, β, c = model
    e = w_vals / (1 - β)
    h = c + β * P @ v
    σ = jnp.where(e >= h, 1, 0)
    return σ
```

Here's a routine for value function iteration.

```{code-cell} ipython3
def vfi(model, max_iter=10_000, tol=1e-4):
    "Solve the infinite-horizon Markov job search model by VFI."
    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
```

## Computing the solution

Let's set up and solve the model.

```{code-cell} ipython3
model = create_js_model()
n, w_vals, P, β, c = model

v_star, σ_star = vfi(model)
```

Here's the optimal policy:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(σ_star)
ax.set_xlabel("wage values")
ax.set_ylabel("optimal choice (stop=1)")
plt.show()
```

We compute the reservation wage as the first $w$ such that $\sigma(w)=1$.

```{code-cell} ipython3
stop_indices = jnp.where(σ_star == 1)
stop_indices
```

```{code-cell} ipython3
res_wage_index = min(stop_indices[0])
```

```{code-cell} ipython3
res_wage = w_vals[res_wage_index]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star, alpha=0.8, label="value function")
ax.vlines((res_wage,), 150, 400, 'k', ls='--', label="reservation wage")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

## Exercise

In the setting above, the agent is risk-neutral vis-a-vis future utility risk.

Now solve the same problem but this time assuming that the agent has risk-sensitive
preferences, which are a type of nonlinear recursive preferences.

The Bellman equation becomes

$$
    v(w) = \max
    \left\{
            \frac{w}{1-\beta}, 
            c + \frac{\beta}{\theta}
            \ln \left[ 
                      \sum_{w'} \exp(\theta v(w')) P(w, w')
                \right]
    \right\}
$$


When $\theta < 0$ the agent is risk averse.

Solve the model when $\theta = -0.1$ and compare your result to the risk neutral
case.

Try to interpret your result.

You can start with the following code:

```{code-cell} ipython3
Model = namedtuple('Model', ('n', 'w_vals', 'P', 'β', 'c', 'θ'))
```

```{code-cell} ipython3
def create_risk_sensitive_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
        θ=-0.1       # risk parameter
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), mc.P
    P = jnp.array(P)
    return Model(n, w_vals, P, β, c, θ)
```

Now you need to modify `T` and `get_greedy` and then run value function iteration again.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for i in range(20):
    print("Solution below!")
```

```{code-cell} ipython3
@jax.jit
def T_rs(v, model):
    """
    The Bellman operator Tv = max{e, c + β R v} with 

        e(w) = w / (1-β) and

        (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

    """
    n, w_vals, P, β, c, θ = model
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)


@jax.jit
def get_greedy_rs(v, model):
    " Get a v-greedy policy."
    n, w_vals, P, β, c, θ = model
    e = w_vals / (1 - β)
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    σ = jnp.where(e >= h, 1, 0)
    return σ



def vfi(model, max_iter=10_000, tol=1e-4):
    "Solve the infinite-horizon Markov job search model by VFI."
    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T_rs(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy_rs(v_star, model)
    return v_star, σ_star



model_rs = create_risk_sensitive_js_model()
n, w_vals, P, β, c, θ = model_rs

v_star_rs, σ_star_rs = vfi(model_rs)
```

Let's plot the results together with the original risk neutral case and see what we get.

```{code-cell} ipython3
stop_indices = jnp.where(σ_star_rs == 1)
res_wage_index = min(stop_indices[0])
res_wage_rs = w_vals[res_wage_index]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star,  alpha=0.8, label="risk neutral $v$")
ax.plot(w_vals, v_star_rs, alpha=0.8, label="risk sensitive $v$")
ax.vlines((res_wage,), 100, 400,  ls='--', color='darkblue', 
          alpha=0.5, label=r"risk neutral $\bar w$")
ax.vlines((res_wage_rs,), 100, 400, ls='--', color='orange', 
          alpha=0.5, label=r"risk sensitive $\bar w$")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

```{code-cell} ipython3

```
