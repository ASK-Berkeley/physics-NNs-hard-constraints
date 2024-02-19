import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray
from math import prod
from functools import partial
from jax.debug import print as jprint

from .moe_utils import MoEConfig, dimension_partition, dimension_unpartition
from ...geometry import Function, Grid, DenseGrid, Dimension
from ...geometry import mesh_utils as mu
from ...types import ModelInput, ModelOutput
from ...pdes import AbstractPDE
from ...utils import distributed, pytree_in_axes


class SpatialTemporalMoE(eqx.Module):
    cfg: MoEConfig
    expert: eqx.Module
    pde: AbstractPDE

    def __init__(self, cfg: MoEConfig, constraint_func: eqx.Module, pde: AbstractPDE):
        self.cfg = cfg
        self.expert = constraint_func
        self.pde = pde

    def __call__(self, rngs: PRNGKeyArray, predicted_solution: Function, initial_condition: Function, pde_params: Function):
        n_experts = prod(self.cfg.num_experts)

        # Compute Residual terms
        # nt x nx x ny x d
        complete_residual_terms = self.pde.compute_pde_residual_terms(
            predicted_solution, pde_params)

        # Partition residuals terms in expert splits
        partitioned_residual_terms = jax.tree_map(partial(
            dimension_partition, experts=self.cfg.num_experts), complete_residual_terms)

        # Shard residual terms across experts
        partitioned_residual_terms = jax.tree_map(
            distributed.shard_array, partitioned_residual_terms)

        # Partition solution across experts
        partitioned_predicted_solution = jax.tree_map(
            partial(dimension_partition, experts=self.cfg.num_experts), predicted_solution)

        # Shard solution
        partitioned_predicted_solution = jax.tree_map(
            distributed.shard_array, partitioned_predicted_solution)

        # Residual values computed at the intitial condition
        # 1 x nx x ny x d
        initial_condition = self.pde.compute_pde_residual_terms(
            initial_condition, mu.initial_condition(pde_params), ic=True)
        # In the case of diffusion-sorption we broadcast the initial condition to each constraint
        if self.pde.name == 'Diffusion-Sorption1D':
            initial_condition = jax.tree_map(
                distributed.replicate_array, initial_condition)

        # In the case of navier stokes, we would like to partition the initial condition
        else:
            # 1 x 64 x 64 x 1
            # Partition the IC terms
            initial_condition = jax.tree_map(partial(
                dimension_partition, experts=self.cfg.num_experts), initial_condition)

            # Shard the IC terms
            initial_condition = jax.tree_map(
                distributed.shard_array, initial_condition)

        # Extract predicted IC
        predicted_initial_condition = mu.initial_condition(predicted_solution)

        # Replicate predicted initial condition
        predicted_initial_condition = jax.tree_map(
            distributed.replicate_array, predicted_initial_condition)

        # Compute global boundary condition terms 
        global_boundary_conditions = self.pde.compute_boundary_condition_terms(
            predicted_solution, pde_params)

        # Replicate global boundary conditions
        global_boundary_conditions = jax.tree_map(
            distributed.replicate_array, global_boundary_conditions)

        # Compute RNG Keys for each expert + shard keys
        constraint_rngs = jax.random.split(rngs, n_experts)
        constraint_rngs = distributed.shard_array(constraint_rngs)

        # Replicate PDE Parameters
        pde_params = jax.tree_map(distributed.replicate_array, pde_params)

        # Run experts
        if self.pde.name == 'Diffusion-Sorption1D':
            out: ModelOutput = jax.vmap(self.expert.precomputed_terms,
                                        in_axes=(0,
                                                 0,
                                                 0,
                                                 None,
                                                 None,
                                                 None,
                                                 None))(
                constraint_rngs,
                partitioned_predicted_solution,
                partitioned_residual_terms,
                predicted_initial_condition,
                global_boundary_conditions,
                initial_condition,
                pde_params
            )
        else:
            out: ModelOutput = jax.vmap(self.expert.precomputed_terms,
                                        in_axes=(0,
                                                 0,
                                                 0,
                                                 None,
                                                 None,
                                                 0,
                                                 None))(
                constraint_rngs,
                partitioned_predicted_solution,
                partitioned_residual_terms,
                predicted_initial_condition,
                global_boundary_conditions,
                initial_condition,  # In this case, we actually do vmap over IC
                pde_params
            )

        # Unpartition the predicted solution
        out.solution = jax.tree_map(
            partial(dimension_unpartition, experts=self.cfg.num_experts), out.solution)
        return out
