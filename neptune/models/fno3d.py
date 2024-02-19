import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List
import equinox as eqx
from jaxtyping import PRNGKeyArray

from .modules import SpectralConv3d
from .modules.activations import Activation_Function
from .model_utils import ModelConfig, split_prng_key


def triple_vmap(f):
    return eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(f)))


def run_conv(conv, x):
    shape = x.shape
    conv_x = jnp.reshape(x, (-1, shape[1] * shape[2] * shape[3]))
    conv_x = conv(conv_x)
    conv_x = jnp.reshape(conv_x, (-1, shape[1], shape[2], shape[3]))
    return conv_x


@dataclass(frozen=True)
class FNO3DConfig(ModelConfig):
    modes1: Optional[List[int]] = None
    modes2: Optional[List[int]] = None
    modes3: Optional[List[int]] = None
    layers: Optional[List[int]] = None
    fc_dim: int = 128
    out_dim: int = 1
    n_components: int = 1
    activation: str = "tanh"
    activate_last_layer: bool = False
    mollifier: Optional[float] = None


class FNO3D(eqx.Module):
    config: FNO3DConfig
    sp_convs: List[SpectralConv3d]
    convs: List[eqx.nn.Conv1d]
    fc0: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    activation_fn: callable

    def __init__(self, cfg: FNO3DConfig, rng_key: PRNGKeyArray, x: jax.Array):
        self.config = cfg
        in_dim = x.shape[-1]
        self.sp_convs = []
        self.convs = []
        for in_size, out_size, mode1, mode2, mode3 in zip(cfg.layers,
                                                    cfg.layers[1:],
                                                    cfg.modes1,
                                                    cfg.modes2,
                                                    cfg.modes3):
            rng_key, init_key = split_prng_key(rng_key)
            self.sp_convs.append(SpectralConv3d(init_key,
                                                in_size,
                                                out_size,
                                                mode1,
                                                mode2,
                                                mode3))
            rng_key, init_key = split_prng_key(rng_key)
            self.convs.append(eqx.nn.Conv1d(in_size,
                                            out_size,
                                            1,
                                            key=init_key))
        rng_key, init_key = split_prng_key(rng_key)
        self.fc0 = eqx.nn.Linear(in_dim,
                                 cfg.layers[0],
                                 key=init_key)
        rng_key, init_key = split_prng_key(rng_key)
        self.fc1 = eqx.nn.Linear(cfg.layers[-1],
                                 cfg.fc_dim,
                                 key=init_key)
        rng_key, init_key = split_prng_key(rng_key)
        self.fc2 = eqx.nn.Linear(
            cfg.fc_dim, cfg.out_dim * cfg.n_basis, key=init_key)
        self.activation_fn = Activation_Function(cfg.activation)

    def __call__(self, rngs: PRNGKeyArray, x: jax.Array) -> jax.Array:
        cfg = self.config
        x = triple_vmap(self.fc0)(x)
        # hidden_dim x nt x nx
        x = jnp.transpose(x, (3, 0, 1, 2))
        for i, (speconv, conv) in enumerate(zip(self.sp_convs, self.convs)):
            # hidden_dim x nt x nx x ny
            x1 = speconv(x)
            # hidden_dim x nt x nx
            x2 = run_conv(conv, x)
            # hidden_dim x nt x n
            x = x1 + x2
            if i != len(self.sp_convs) - 1:
                x = self.activation_fn(x)
        # nt x nx x hidden_dim
        x = jnp.transpose(x, (1, 2, 3, 0))
        x = self.activation_fn(triple_vmap(self.fc1)(x))
        x = triple_vmap(self.fc2)(x)
        if cfg.activate_last_layer:
            x = self.activation_fn(x)

        # nt, nx, out_dim * n_basis
        if cfg.n_basis > 1:
            x = jnp.reshape(
                x, x.shape[:-1] + (cfg.n_basis, cfg.out_dim))
        else:
            x = jnp.reshape(x, x.shape[:-1] + (cfg.out_dim,))
        return x