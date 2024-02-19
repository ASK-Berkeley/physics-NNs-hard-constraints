import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray


def is_2d(u):
    return len(jnp.shape(u)) == 3


def is_3d(u):
    return len(jnp.shape(u)) == 4


class ICMollifier(eqx.Module):
    scaling: float
    subtract_mean: bool

    def __init__(self, scaling: float, subtract_mean: bool):
        self.scaling = scaling
        self.subtract_mean = subtract_mean

    def __call__(self, rngs: PRNGKeyArray, u: jax.Array, t: jax.Array = None, ic: jax.Array = None) -> jax.Array:
        del rngs  # unused

        # No mollifier
        if self.scaling is None:
            return u

        if ic is None:  # Multiplication by t
            u *= self.scaling * t / t[-1:]
        else:  # Addition of IC
            u = u + ic

        if self.subtract_mean and is_2d(u) and ic is None:
            u -= jnp.mean(u, axis=(1), keepdims=True)
        elif self.subtract_mean and is_3d(u):
            u -= jnp.mean(u, axis=(1, 2), keepdims=True)
        return u
