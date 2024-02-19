import jax.numpy as jnp
import tensorflow as tf
import jax
import typing
from functools import partial
from .abstract_pde import AbstractPDE
from ..geometry import SparseGrid, Function
from ..geometry import mesh_utils as mu
from ..types import ModelInput
from .finite_differences import fdm_first_order_derivative


class DiffusionSorption1D(AbstractPDE):
    name: str = 'Diffusion-Sorption1D'

    def tf_schema(self):
        return super()._tf_schema(['D', 'phi', 'rho_s', 'n_f', 'k', 'u', 'x', 't'])

    def reweight_matrix(self, weight, matrix):
        out = matrix @ weight
        out = jnp.expand_dims(out, axis=-1)
        return out

    def broadcast_pde_params(self, D, phi, rho_s, n_f, k, grid_shape):
        # scalars are 0 dim tensors
        pde_params = [D, phi, rho_s, n_f, k]
        # n_param
        pde_params = jnp.asarray(pde_params)
        pde_params = jnp.squeeze(pde_params)
        # 1 x n_param
        pde_params = jnp.expand_dims(pde_params, axis=0)
        # 1 x 1 x n_param
        pde_params = jnp.expand_dims(pde_params, axis=0)
        # nx x nt x n_param
        pde_params = jnp.broadcast_to(
            pde_params, grid_shape + (pde_params.shape[-1],))
        return pde_params

    def process_input(self, input_dict):
        # Scalars
        D = input_dict['D']
        phi = input_dict['phi']
        rho_s = input_dict['rho_s']
        n_f = input_dict['n_f']
        k = input_dict['k']

        # Tensors
        # nt x nx x ndim
        u_np = input_dict['u']
        u = u_np
        x = input_dict['x']
        x_normalized = x / x.max()
        t = input_dict['t']
        t_normalized = t / t.max()
        grid = SparseGrid(t, x)
        grid = mu.to_dense(grid)
        grid_normalized = SparseGrid(t_normalized, x_normalized)
        grid_normalized = mu.to_dense(grid_normalized)

        if len(jnp.shape(u)) == 2:
            # Ensure that u is a 3D tensor
            u = jnp.expand_dims(u, axis=-1)

        pde_sol = Function(grid_normalized, u)
        # nt x nx x n_param
        pde_params = self.broadcast_pde_params(D=D,
                                               phi=phi,
                                               rho_s=rho_s,
                                               n_f=n_f,
                                               k=k,
                                               grid_shape=grid.shape[:-1])
        pde_params = Function(grid, pde_params)
        domain = grid_normalized
        initial_condition = mu.initial_condition(pde_sol)
        model_input = ModelInput(
            domain=domain,
            pde_param=pde_params,
            initial_condition=initial_condition,
        )
        return model_input, pde_sol

    def compute_boundary_condition_terms(self, u: Function, pde_params: Function) -> typing.Dict[str, jax.Array]:
        domain = u.domain.grid
        u = u.image
        u_dx = fdm_first_order_derivative(u, domain, axis=1)
        return dict(left=u[1:, 0], right_u=u[1:, -1], right_dx=u_dx[1:, -1])

    def compute_boundary_conditions(self, terms: typing.Dict[str, jax.Array]):
        ### LEFT BC LOSS u(t, 0) = 1.0 ###
        left_u = terms['left']
        left_boundary_condition = left_u - 1.

        ### RIGHT BC LOSS u(t, 1) = D dx u(t, 1) ###
        right_u, right_u_dx = terms['right_u'], terms['right_dx']
        D = 5e-4
        right_boundary_condition = right_u - D * right_u_dx

        return (left_boundary_condition, right_boundary_condition)

    def compute_pde_residual_terms(self, u: Function, pde_params: Function, ic: bool = False):
        domain = pde_params.domain.grid
        u = u.image
        if not ic:
            u_dt = fdm_first_order_derivative(u, domain, axis=0)
        else:
            u_dt = jnp.zeros_like(u)

        u_dx = fdm_first_order_derivative(u, domain, axis=1)
        u_dxx = fdm_first_order_derivative(u_dx, domain,  axis=1)
        return dict(u=u, u_dx=u_dx, u_dxx=u_dxx, u_dt=u_dt)

    def compute_pde_residual(self, terms: typing.Dict[str, jax.Array]):
        D: float = 5e-4
        por: float = 0.29
        rho_s: float = 2880
        k_f: float = 3.5e-4
        n_f: float = 0.874

        u = terms['u']
        u_dxx = terms['u_dxx']
        u_dt = terms['u_dt']

        safe_u = jnp.where(u > 0, u, 0)
        retardation_factor = 1 + ((1 - por) / por) * \
            rho_s * k_f * n_f * \
            jnp.where(u > 0, jnp.power(safe_u + 1e-6, n_f - 1), 1e-6**(n_f-1))
        lhs = u_dt - D / retardation_factor * u_dxx
        rhs = jnp.zeros_like(lhs)
        return lhs, rhs

    def vector_objective_function(self,
                                  weight: jax.Array,
                                  args):
        _, sampled_residual_terms, initial_condition, predicted_initial_condition, boundary_condition_terms = args

        # Reweight relevant terms
        sampled_residual_terms = jax.tree_map(
            partial(self.reweight_matrix, weight), sampled_residual_terms)
        predicted_initial_condition = self.reweight_matrix(
            weight, predicted_initial_condition)
        boundary_condition_terms = jax.tree_map(
            partial(self.reweight_matrix, weight), boundary_condition_terms)

        ### RESIDUAL LOSS ###
        residual_lhs, residual_rhs = self.compute_pde_residual(
            sampled_residual_terms)
        residual_lhs, residual_rhs = residual_lhs.ravel(), residual_rhs.ravel()
        residual_loss = residual_lhs - residual_rhs
        residual_loss = residual_loss.ravel()

        ### IC LOSS ###
        ic_loss = predicted_initial_condition - initial_condition['u']
        ic_loss = ic_loss.ravel() / initial_condition['u'].ravel()

        ### BOUNDARY LOSS ###
        boundary_conditions = self.compute_boundary_conditions(
            boundary_condition_terms)
        boundary_conditions = jax.tree_map(
            lambda x: x.ravel(), boundary_conditions)

        ### RIDGE REGULARIZATION ###
        ridge_weight = 1e-4
        ridge_loss = ridge_weight * weight.ravel()

        total_loss_terms = jnp.concatenate((
            residual_loss,
            ic_loss,
            *boundary_conditions,
            ridge_loss))

        return total_loss_terms
