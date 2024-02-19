import jax
import jax.numpy as jnp
import typing
import equinox as eqx
from functools import partial
from jaxtyping import PRNGKeyArray
import optimistix as optx
import jaxopt

from .constraint_utils import ConstraintConfig
from ...geometry import Function, generate_single_mask, apply_mask, Mask
from ...geometry import mesh_utils as mu
from ...types import ModelOutput
from ...pdes import AbstractPDE


def reweight_matrix(weight, matrix):
    out = matrix @ weight
    out = jnp.expand_dims(out, axis=-1)
    return out


def setup_hard_constraint_layer(cfg: ConstraintConfig, pde: AbstractPDE):
    if cfg.system == 'levenbergmarquardt':
        if cfg.use_jaxopt:
            solver = jaxopt.LevenbergMarquardt
            objective = None
        else:
            solver = optx.LevenbergMarquardt(rtol=cfg.rtol,
                                         atol=cfg.atol)
            objective = pde.vector_objective_function
    if cfg.system == 'bfgs':
        pass
    if cfg.system == 'gaussnewton':
        pass
    return solver, objective


class HardConstraintLayer(eqx.Module):
    cfg: ConstraintConfig
    pde: AbstractPDE
    objective_func: typing.Callable

    def __init__(self, cfg: ConstraintConfig, pde: AbstractPDE):
        self.cfg = cfg
        self.pde = pde
        self.objective_func = pde.vector_objective_function
        self.objective_func = lambda w, args, mask: pde.vector_objective_function(w, args, mask, self.cfg.num_sampled_points)

    def __call__(self, rngs: PRNGKeyArray, u: Function, pde_params: Function, initial_condition: Function):
        ### Case where the hard constraint is used without MoE ###
        pde = self.pde

        # Compute individual terms in the PDE residual
        residual_terms = pde.compute_pde_residual_terms(u, pde_params)

        # Compute the boundary condition terms
        boundary_conditions = pde.compute_boundary_condition_terms(
            u, pde_params)

        # Extract predicted IC
        predicted_initial_condition = mu.initial_condition(u)

        # Compute IC terms
        initial_condition = pde.compute_pde_residual_terms(
            initial_condition, mu.initial_condition(pde_params), ic=True)

        initial_condition = jax.tree_map(
            lambda x: jnp.broadcast_to(
                x, (u.shape[0], *x.shape[1:])), initial_condition
        )


        return self.precomputed_terms(rngs=rngs,
                                      u=u,
                                      residual_terms=residual_terms,
                                      predicted_initial_condition=predicted_initial_condition,
                                      boundary_conditions=boundary_conditions,
                                      initial_condition=initial_condition,
                                      pde_params=pde_params)

    def precomputed_terms(self, rngs: PRNGKeyArray, u: Function, residual_terms: typing.Dict[str, jax.Array],
                          predicted_initial_condition: Function, boundary_conditions: typing.Dict[str, jax.Array],
                          initial_condition: typing.Dict[str, jax.Array], pde_params: Function):
        cfg = self.cfg
        pde = self.pde

        # Setup the NLLS Solver and corresponding objective function
        solver, objective = setup_hard_constraint_layer(cfg, pde)
        if cfg.use_jaxopt:
            solver = solver(self.objective_func, tol=cfg.tol, maxiter=cfg.maxiter)
            

        # Mask out and sample N points from the domain applied to each individual term
        rngs, subkey = jax.random.split(rngs)
        mask = generate_single_mask(u.domain, cfg.num_sampled_points, subkey)
        sampled_residual_terms = jax.tree_map(
            partial(apply_mask, mask=mask), residual_terms)

        if cfg.mask_boundary_conditions: 
            # NS Case
            sampled_residual_terms = residual_terms

        return self.solve(rngs=rngs,
                          solver=solver,
                          objective=objective,
                          u=u,
                          residual_terms=sampled_residual_terms,
                          initial_condition=initial_condition,
                          predicted_initial_condition=predicted_initial_condition,
                          boundary_condition_terms=boundary_conditions,
                          mask=mask,
                          mask_size=cfg.num_sampled_points)

    def solve(self, rngs: PRNGKeyArray, solver: optx._solver, objective: typing.Callable, u: Function,
              residual_terms: typing.Dict[str, jax.Array], initial_condition: typing.Dict[str, jax.Array], predicted_initial_condition: Function,
              boundary_condition_terms: typing.Dict[str, jax.Array], mask, mask_size):
        # Instantiate the solver and solve the NLLS problem
        if self.pde.name == 'Diffusion-Sorption1D':
            w_init = jnp.ones(u.image.shape[-1])
        else:
            w_init = jnp.zeros(u.image.shape[-1])
        if self.cfg.use_jaxopt:
            optstate = solver.run(w_init, (u, residual_terms, initial_condition,
                                  predicted_initial_condition.image, boundary_condition_terms), mask.mask)
            solved_w = optstate.params
        else:
            soln = optx.least_squares(objective, solver, w_init, args=(u,
                                                            residual_terms,
                                                            initial_condition,
                                                            predicted_initial_condition.image,
                                                            boundary_condition_terms,
                                                            ),
                            max_steps=self.cfg.maxiter,
                            throw=False)
            solved_w = soln.value
        u = Function(u.domain, self.pde.reweight_matrix(
            solved_w, u.image))
        if self.cfg.use_jaxopt:
            return ModelOutput(solution=u,
                           weight=solved_w,
                           solver_iter=optstate.state.iter_num,
                           solver_status=optstate.state.error)
        else:
            return ModelOutput(solution=u,
                           weight=solved_w,
                           solver_iter=soln.stats['num_steps'],
                           solver_status=soln.result)
