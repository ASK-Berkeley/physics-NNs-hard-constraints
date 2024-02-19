import jax.numpy as jnp
import jax
import typing
from typing import Dict
from functools import partial
from neptune.geometry import Function

from .abstract_pde import AbstractPDE
from ..geometry import SparseGrid, Function
from ..geometry import mesh_utils as mu
from ..types import ModelInput


class NavierStokes2D(AbstractPDE):
    name: str = 'Navier-Stokes2D'

    def tf_schema(self):
        return super()._tf_schema(['u', 't', 'x', 'y', 'reynolds_no'])

    def broadcast_pde_params(self, reynolds_no, grid_shape):
        # scalars are 0 dim tensors
        pde_params = [reynolds_no]
        # n_param
        pde_params = jnp.asarray(pde_params)
        if len(pde_params.shape) > 1:
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
        reynolds_no = input_dict['reynolds_no']

        # Tensors
        u_np = input_dict['u']
        if len(u_np.shape) == 3:
            u_np = jnp.expand_dims(u_np, axis=-1)
        t = input_dict['t']
        x = input_dict['x']
        y = input_dict['y']
        grid = SparseGrid(t, x, y)
        grid = mu.to_dense(grid)

        # Sol: nt x nx x ny x 2
        pde_sol = Function(grid, u_np)
        # nt x nx x n_param
        pde_params = self.broadcast_pde_params(reynolds_no=reynolds_no,
                                               grid_shape=grid.shape[:-1])
        pde_params = Function(grid, pde_params)
        domain = grid
        initial_condition = mu.initial_condition(pde_sol)
        model_input = ModelInput(
            domain=domain,
            pde_param=pde_params,
            initial_condition=initial_condition,
        )
        return model_input, pde_sol

    def compute_boundary_condition_terms(self, u: Function, pde_params: Function) -> Dict[str, jax.Array]:
        return dict(ux=jnp.zeros_like(u.image))

    def compute_boundary_conditions(self, terms: Dict[str, jax.Array]):
        # No boundary conditions, so we just return an array of zeros
        return jnp.zeros_like(terms['ux'])

    def forcing_function(self, y):
        nt, nx, ny, _ = y.shape
        return jnp.zeros((nt, nx, ny))

    def compute_pde_residual_terms(self, u: Function, pde_params: Function, ic: bool = False):
        w = u.image
        nt, nx, ny, _ = w.shape

        # Wavenumbers in x and y-direction
        kx = jnp.tile(jnp.fft.fftfreq(nx)[:, None] * nx * 2*jnp.pi, (1, ny))
        ky = jnp.tile(jnp.fft.fftfreq(ny)[None, :] * ny * 2*jnp.pi, (nx, 1))

        # Negative Laplacian
        Δ = kx ** 2 + ky ** 2
        Δ = Δ.at[0, 0].set(1)

        def fdm(x, n: int = 2):
            u = [x]
            d = x.ndim - 1
            s = x.shape[:-1]

            for _ in range(n):

                grad = map(lambda i: jnp.gradient(x, axis=i), range(d))
                u.append(x := jnp.stack(tuple(grad), axis=-1) * jnp.array(s))

            return u

        def velocity(what=None, *, w=None):

            if what is None:
                what = jnp.fft.fft2(w)

            vx = jnp.fft.irfft2(what * 1j*ky / Δ, what.shape)
            vy = jnp.fft.irfft2(what * -1j*kx / Δ, what.shape)

            return vx, vy

        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def _compute_pde_residual_term(w: jax.Array):
            _, w1, w2 = jax.vmap(fdm)(w[..., None])

            wx = w1[..., 0, 0]
            wy = w1[..., 0, 1]

            wlap = jnp.einsum("...ii -> ...", w2[..., 0, :, :])
            ux, uy = jax.vmap(velocity)(w=w)

            return dict(ux=ux, uy=uy, wx=wx, wy=wy, wlap=wlap)
        if ic:
            return _compute_pde_residual_term(w)

        t = u.domain.grid[..., 0]
        dt = (t.max() - t.min()) / (nt-1)

        lhs = _compute_pde_residual_term(w)
        lhs["wt"] = jnp.gradient(w, axis=0) / dt

        rhs = jnp.zeros_like(w)
        return lhs, rhs

    def compute_pde_residual_lhs(self, lhs: Dict[str, jax.Array],
                                 Re: float) -> jax.Array:

        Dw = lhs["ux"] * lhs["wx"] + lhs["uy"] * lhs["wy"]
        return lhs["wt"] + Dw - lhs["wlap"] / Re

    def compute_pde_residual(self, terms: typing.Tuple[Dict[str, jax.Array]]) -> typing.Tuple[jax.Array]:
        Re = 1e4   
        lhs, rhs = terms
        lhs = self.compute_pde_residual_lhs(lhs, Re)
        return lhs, rhs

    @partial(jax.jit, static_argnames=['self', 'mask_size'])
    def vector_objective_function(self,
                                  weight: jax.Array,
                                  args, mask, mask_size):
        sol, residual_terms, ic, predicted_initial_condition, boundary_condition_terms = args
        lhs, rhs = residual_terms
        lhs = jax.tree_map(partial(self.reweight_matrix, weight), lhs)

        lhs["ux"] += ic["ux"]
        lhs["uy"] += ic["uy"]
        lhs["wx"] += ic["wx"]
        lhs["wy"] += ic["wy"]
        lhs["wlap"] += ic["wlap"]

        lhs, rhs = self.compute_pde_residual(terms=(lhs, rhs))
        idx = jnp.argwhere(mask.ravel(), size=mask_size)
        lhs, rhs = lhs.ravel()[idx], rhs.ravel()[idx]

        ridge_weight = 1e-4
        res = jnp.square(lhs-rhs).ravel()

        return jnp.concatenate([jnp.square(lhs - rhs).ravel(),
                                ridge_weight * weight.ravel()])
