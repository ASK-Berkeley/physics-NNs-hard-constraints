import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray

"""
Adapted from https://github.com/neural-operator/fourier_neural_operator
"""


def compl_mul3d(a: jax.Array, b: jax.Array):
    # ( in_channel, t, x, y), (in_channel, out_channel, t,x,y) -> (out_channel,t,x,y)
    return jnp.einsum("ixyz,ioxyz->oxyz", a, b)


def initializer_3d(key,
                   in_channels,
                   out_channels,
                   mode1,
                   mode2,
                   mode3):
    scale = 1 / (in_channels * out_channels)
    key_real, key_imaginary, key = jax.random.split(key, 3)
    real = jax.random.uniform(
        key_real,
        (
            4,
            in_channels,
            out_channels,
            mode1,
            mode2,
            mode3,
        ),
    )
    imaginary = jax.random.uniform(
        key_imaginary,
        (
            4,
            in_channels,
            out_channels,
            mode1,
            mode2,
            mode3,
        ),
    )
    return scale * jax.lax.complex(real, imaginary)


class SpectralConv3d(eqx.Module):
    in_channels: int
    out_channels: int
    mode1: int
    mode2: int
    mode3: int
    weights1: jax.Array
    weights2: jax.Array
    weights3: jax.Array
    weights4: jax.Array

    def __init__(self, rng: PRNGKeyArray,
                 in_channels: int,
                 out_channels: int,
                 mode1: int,
                 mode2: int,
                 mode3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode1 = mode1
        self.mode2 = mode2
        self.mode3 = mode3
        weights = initializer_3d(rng,
                                 in_channels,
                                 out_channels,
                                 mode1,
                                 mode2,
                                 mode3)
        self.weights1 = weights[0]
        self.weights2 = weights[1]
        self.weights3 = weights[2]
        self.weights4 = weights[3]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: n_dim, nt, nx, ny
        # Compute Fourier coeffcients up to factor of e^(-constant)
        x_ft = jnp.fft.rfftn(x, axes=[1, 2, 3])

        z_dim = min(x_ft.shape[3], self.mode3)

        # Multiply relevant Fourier modes
        # b x c x t x X x y -> b x c x t x X x y
        out_ft = jnp.zeros((self.out_channels,
                           x_ft.shape[1],
                           x_ft.shape[2],
                           self.mode3), dtype=jnp.complex_)

        # if x_ft.shape[4] > self.mode3, truncate
        # if x_ft.shape[4] < self.mode3, add zero padding
        coeff = jnp.zeros((self.in_channels,
                          self.mode1,
                          self.mode2,
                          self.mode3),
                          dtype=jnp.complex_)
        coeff = coeff.at[..., :z_dim].set(
            x_ft[..., :self.mode1, :self.mode2, :z_dim])
        out_ft = out_ft.at[..., :self.mode1, :self.mode2, :].set(
            compl_mul3d(coeff, self.weights1))

        coeff = jnp.zeros((self.in_channels,
                          self.mode1,
                          self.mode2,
                          self.mode3),
                          dtype=jnp.complex_)
        coeff = coeff.at[..., :z_dim].set(
            x_ft[..., -self.mode1:, :self.mode2, :z_dim])
        out_ft = out_ft.at[..., -self.mode1:, :self.mode2,
                           :].set(compl_mul3d(coeff, self.weights2))

        coeff = jnp.zeros((self.in_channels,
                          self.mode1,
                          self.mode2,
                          self.mode3),
                          dtype=jnp.complex_)
        coeff = coeff.at[..., :z_dim].set(
            x_ft[..., :self.mode1, -self.mode2:, :z_dim])
        out_ft = out_ft.at[..., :self.mode1, -self.mode2:,
                           :].set(compl_mul3d(coeff, self.weights3))

        coeff = jnp.zeros((self.in_channels,
                          self.mode1,
                          self.mode2,
                          self.mode3),
                          dtype=jnp.complex_)
        coeff = coeff.at[..., :z_dim].set(
            x_ft[..., -self.mode1:, -self.mode2:, :z_dim])
        out_ft = out_ft.at[..., -self.mode1:, -self.mode2:,
                           :].set(compl_mul3d(coeff, self.weights4))
        x = jnp.fft.irfftn(out_ft, s=(
            x.shape[1], x.shape[2], x.shape[3]), axes=[1, 2, 3])
        return x
