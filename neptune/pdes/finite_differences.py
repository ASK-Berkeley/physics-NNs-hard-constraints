import jax
import jax.numpy as jnp


def fdm_first_order_derivative(u, domain, axis):
    df_dx_interior = central_fdm(u, domain, axis)
    # We only need to compute the boundary points
    df_dx_lower = forward_fdm(jax.lax.slice_in_dim(u, 0, 2, axis=axis),
                              jax.lax.slice_in_dim(domain, 0, 2, axis=axis),
                              axis=axis)
    df_dx_upper = backward_fdm(jax.lax.slice_in_dim(u, -2, None, axis=axis),
                               jax.lax.slice_in_dim(
                                   domain, -2, None, axis=axis),
                               axis=axis)
    df_dx = jnp.concatenate(
        (df_dx_lower, df_dx_interior, df_dx_upper), axis=axis)
    return df_dx


def central_fdm(image: jax.Array, domain: jax.Array, axis: int):
    # (1 / 2dx) * (f(x + dx) - f(x - dx))
    # domain: nt x nx ... x ndim
    # image: nt x nx ... x outdim
    x = domain[..., axis]
    x = jnp.expand_dims(x, -1)
    lower_x = jax.lax.slice_in_dim(x, 0, -2, axis=axis)
    upper_x = jax.lax.slice_in_dim(x, 2, None, axis=axis)
    dx = upper_x - lower_x
    f_lower = jax.lax.slice_in_dim(image, 0, -2, axis=axis)
    f_upper = jax.lax.slice_in_dim(image, 2, None, axis=axis)
    # Notation-wise, typical writeups use 2 * dx. We use dx since our dx = 2 * grid_spacing
    df_dx_interior = (1 / dx) * (f_upper - f_lower)
    return df_dx_interior


def forward_fdm(image: jax.Array, domain: jax.Array, axis: int):
    x = domain[..., axis]
    x = jnp.expand_dims(x, -1)
    x_upper = jax.lax.slice_in_dim(x, 1, None, axis=axis)
    x = jax.lax.slice_in_dim(x, 0, -1, axis=axis)
    dx = x_upper - x
    f = jax.lax.slice_in_dim(image, 0, -1, axis=axis)
    f_upper = jax.lax.slice_in_dim(image, 1, None, axis=axis)
    df_dx_lower = (1 / dx) * (f_upper - f)
    return df_dx_lower


def backward_fdm(image: jax.Array, domain: jax.Array, axis: int):
    x = domain[..., axis]
    x = jnp.expand_dims(x, -1)
    x_lower = jax.lax.slice_in_dim(x, None, -1, axis=axis)
    x = jax.lax.slice_in_dim(x, 1, None, axis=axis)
    dx = x - x_lower
    f = jax.lax.slice_in_dim(image, 1, None, axis=axis)
    f_lower = jax.lax.slice_in_dim(image, 0, -1, axis=axis)
    df_dx_upper = (1 / dx) * (f - f_lower)
    return df_dx_upper
