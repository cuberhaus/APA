import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, random


@jax.jit
def myfunction(p, x):
    return jnp.sum(p * jnp.array([jnp.cos(x), jnp.sin(3 * x)]).T)


@jax.jit
def lossMAE(p, x, y):
    return jnp.abs(myfunction(p, x) - y)


@jax.jit
def lossMSE(p, x, y):
    return (myfunction(p, x) - y) ** 2

