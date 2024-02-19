import jax
import jax.numpy as jnp
from functools import partial
import math


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def gaussian_rf2d(key, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.5, tau=3.0, sigma=None, mean=None):
    if sigma is None:
        sigma = tau ** (0.5 * (2 * alpha - 2.0))
    c1 = (4 * (math.pi ** 2)) / (L1 ** 2)
    c2 = (4 * (math.pi ** 2)) / (L2 ** 2)
    freq1 = jnp.concatenate(
        (jnp.arange(s1//2), jnp.arange(start=-s1//2, stop=0)))
    k1 = jnp.tile(freq1.reshape(-1, 1),
                  (1, s2//2))
    freq2 = jnp.arange(s2//2)
    k2 = jnp.tile(freq2.reshape(1, -1),
                  (s1, 1))
    sqrt_eig = s1 * s2 * sigma * \
        ((c1 * k1 ** 2 + c2 * k2 ** 2 + tau ** 2) ** (-alpha / 2.0))
    sqrt_eig = sqrt_eig.at[0, 0].set(0.0)

    # sample
    xi = jax.random.normal(key, (s1, s2//2, 2))
    xi = xi.at[..., 0].set(sqrt_eig * xi[..., 0])
    xi = xi.at[..., 1].set(sqrt_eig * xi[..., 1])
    xi = jax.lax.complex(xi[..., 0], xi[..., 1])

    u = jnp.fft.irfft2(xi, s=(s1, s2))
    if mean is not None:
        u += mean
    return u
