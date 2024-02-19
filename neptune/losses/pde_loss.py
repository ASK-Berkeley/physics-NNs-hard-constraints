import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from .mse import MSE
from ..geometry import Function, Grid
from ..geometry import mesh_utils as mu
from ..types import class_type
from jax.debug import print as jprint


class PDELoss(eqx.Module):
    """
    Loss for PDEs
    """
    pde: class_type
    mse: MSE

    def __init__(self, pde):
        self.pde = pde
        self.mse = MSE(normalize=False)

    def single_pde(self, predicted: Function, pde_param: Function):
        terms = self.pde.compute_pde_residual_terms(predicted, pde_param)
        lhs, rhs = self.pde.compute_pde_residual(terms)
        return jnp.linalg.norm((lhs - rhs).ravel())

    def __call__(self, predicted: Function,
                 pde_param: Function):
        return jax.vmap(self.single_pde)(predicted, pde_param).mean()
