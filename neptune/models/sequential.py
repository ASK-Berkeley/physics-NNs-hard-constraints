import jax
import jax.numpy as jnp
import equinox as eqx
import typing
import jax
from functools import partial
from jaxtyping import PRNGKeyArray
from jax.debug import print as jprint
from .modules import ICMollifier
from ..types import ModelInput, ModelOutput
from ..geometry import Function, DenseGrid
from .MoE import SpatialTemporalMoE, MoEConfig
from ..pdes import AbstractPDE


class SequentialModel(eqx.Module):
    """
    Wraps learnable parameters and constraints.
    """
    model: eqx.Module
    constraint: eqx.Module
    MoE: eqx.Module

    def __init__(self, model: eqx.Module, constraint: eqx.Module, MoE: MoEConfig = None, pde: AbstractPDE = None):
        self.model = model
        self.constraint = constraint
        if MoE is not None and MoE.split == 'spatialtemporal':
            self.MoE = SpatialTemporalMoE(MoE, constraint, pde)
        else:
            self.MoE = None

    def prepare_input(self, input_data: ModelInput) -> jax.Array:
        # Generates the model input by broadcasting the initial condition and concatenating it with the domain
        # initial_condition: nx x ny ... x out_dim
        # pde_params: nt x nx x ... x n_params
        # domain: nt x nx x ... x n_spatial
        initial_condition = input_data.initial_condition.image
        pde_params = input_data.pde_param.image  # unused
        domain = input_data.domain

        initial_condition = jnp.broadcast_to(initial_condition,
                                             (domain.shape[0], *initial_condition.shape[1:]))
        grid = domain.grid
        x = jnp.concatenate(
            (grid, initial_condition), axis=-1)
        return x

    def __call__(self, rngs: PRNGKeyArray, input_data: ModelInput, pde_sol: Function) -> ModelOutput:
        # Runs the model and applies the constraint if it exists
        model_cfg = self.model.config
        x = jax.vmap(self.prepare_input)(input_data)
        model_out: jax.Array = jax.vmap(self.model)(rngs, x)
        model_out: ModelOutput = ModelOutput(
            solution=Function(input_data.domain, model_out)
        )

        use_mollifier = hasattr(self.model.config, 'mollifier') and self.model.config.mollifier is not None
        if use_mollifier:
            model_out = ModelOutput(solution=basis_mollifier(self.model.config.mollifier, model_out.solution.image, model_out.solution.domain))

        if self.MoE is None and self.constraint is not None:
            model_out = jax.vmap(self.constraint)(rngs['sampler'],
                                                  model_out.solution,
                                                  input_data.pde_param,
                                                  input_data.initial_condition)
        elif self.MoE is not None:
            model_out = jax.vmap(self.MoE)(rngs['sampler'],
                                           model_out.solution,
                                           input_data.initial_condition,
                                           input_data.pde_param)
            
        if use_mollifier:
            model_out.solution.image += input_data.initial_condition.image
            model_out.solution.image -= jnp.mean(model_out.solution.image, axis=(2, 3), keepdims=True)

        return model_out

@partial(jax.vmap, in_axes=(None, 0, 0))
def basis_mollifier(scaling: float, predicted: jax.Array, domain: DenseGrid):
    predicted *= scaling * (t := domain.grid[..., :1]) / t[-1:]
    predicted -= jnp.mean(predicted, axis=(1, 2), keepdims=True)
    return Function(domain, predicted)