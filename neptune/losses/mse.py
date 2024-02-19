import jax
import jax.numpy as jnp
from typing import Optional
import equinox as eqx


def lp_norm(x, ord):
    return jnp.linalg.norm(x.ravel(), ord=ord)


class MSE(eqx.Module):
    """
    Mean squared error
    """
    normalize: bool = True
    ord: int = 2
    square_root: bool = False

    def __call__(self, pred: jax.Array, true: jax.Array):
        loss = jnp.linalg.norm(jnp.ravel(pred - true))

        return loss if not self.normalize else \
               loss / jnp.linalg.norm(jnp.ravel(true))
