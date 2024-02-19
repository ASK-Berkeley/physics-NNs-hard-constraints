import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PRNGKeyArray

"""
Adapted from https://github.com/neural-operator/fourier_neural_operator
"""


def compl_mul2d(a: jax.Array, b: jax.Array):
    # ( in_channel, t, x), (in_channel, out_channel, t,x) -> (out_channel,t,x)
    return jnp.einsum("itx,iotx->otx", a, b)


def initializer_2d(key: PRNGKeyArray,
                   in_channels: int,
                   out_channels: int,
                   mode1: int,
                   mode2: int) -> jax.Array:
    fan_in = in_channels
    fan_out = out_channels
    scale = 1 / jnp.sqrt((fan_in + fan_out) / 2)
    key_real, key_imaginary, key = jax.random.split(key, 3)
    real = jax.random.normal(
        key_real,
        (
            2,
            in_channels,
            out_channels,
            mode1,
            mode2,
        ),
    )
    imaginary = jax.random.normal(
        key_imaginary,
        (
            2,
            in_channels,
            out_channels,
            mode1,
            mode2,
        ),
    )
    return scale * jax.lax.complex(real, imaginary)


class SpectralConv2d(eqx.Module):
    in_channels: int
    out_channels: int
    mode1: int
    mode2: int
    weights1: jax.Array
    weights2: jax.Array

    def __init__(self, rng: PRNGKeyArray,
                 in_channels: int,
                 out_channels: int,
                 mode1: int,
                 mode2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode1 = mode1
        self.mode2 = mode2
        weights = initializer_2d(rng, in_channels, out_channels, mode1, mode2)
        self.weights1 = weights[0]
        self.weights2 = weights[1]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: n_dim, nt, nx
        # Compute Fourier coeffcients up to factor of e^(-constant)
        in_dim = x.shape[0]
        nt = x.shape[1]
        nx = x.shape[2]

        # x_ft: n_dim, nt, nx
        x_ft = jnp.fft.rfftn(x, axes=[1, 2])

        # Multiply relevant Fourier modes
        out_ft = jnp.zeros(
            (self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1),
            dtype=jnp.complex_,
        )
        # print('out_ft', out_ft.shape)
        out_ft = out_ft.at[:, : self.mode1, : self.mode2].set(
            compl_mul2d(x_ft[:, : self.mode1, : self.mode2], self.weights1)
        )
        out_ft = out_ft.at[:, -self.mode1:, : self.mode2].set(
            compl_mul2d(x_ft[:, -self.mode1:, : self.mode2], self.weights2)
        )

        # Return to physical space
        x = jnp.fft.irfftn(out_ft, s=(
            nt, nx), axes=[1, 2])
        return x
