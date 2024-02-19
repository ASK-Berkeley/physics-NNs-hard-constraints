import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, List
import equinox as eqx
from jaxtyping import PRNGKeyArray

from .modules import SpectralConv2d
from .modules.activations import Activation_Function
from .model_utils import ModelConfig, split_prng_key


def double_vmap(x): return jax.vmap(
    jax.vmap(x, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)


@dataclass(frozen=True)
class FNO2DConfig(ModelConfig):
    modes1: Optional[List[int]] = None
    modes2: Optional[List[int]] = None
    layers: Optional[List[int]] = None
    fc_dim: int = 128
    out_dim: int = 1
    activation: str = "tanh"
    activate_last_layer: bool = False


class FNO2D(eqx.Module):
    config: FNO2DConfig
    sp_convs: List[SpectralConv2d]
    convs: List[eqx.nn.Conv1d]
    fc0: eqx.nn.Linear
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    activation_fn: callable

    def __init__(self, cfg: FNO2DConfig, rng_key: PRNGKeyArray, x: jax.Array):
        self.config = cfg
        in_dim = x.shape[-1]
        self.sp_convs = []
        self.convs = []
        for in_size, out_size, mode1, mode2 in zip(cfg.layers, cfg.layers[1:], cfg.modes1, cfg.modes2):
            rng_key, init_key = split_prng_key(rng_key)
            self.sp_convs.append(SpectralConv2d(init_key,
                                                in_size,
                                                out_size,
                                                mode1,
                                                mode2))
            rng_key, init_key = split_prng_key(rng_key)
            self.convs.append(eqx.nn.Conv2d(in_size,
                                            out_size,
                                            kernel_size=1,
                                            stride=1,
                                            key=init_key))

        rng_key, init_key = split_prng_key(rng_key)
        self.fc0 = double_vmap(eqx.nn.Linear(
            in_dim, cfg.layers[0], key=init_key))
        rng_key, init_key = split_prng_key(rng_key)
        self.fc1 = double_vmap(eqx.nn.Linear(
            cfg.layers[-1], cfg.fc_dim, key=init_key))
        rng_key, init_key = split_prng_key(rng_key)
        self.fc2 = double_vmap(eqx.nn.Linear(
            cfg.fc_dim, cfg.out_dim, key=init_key))
        self.activation_fn = Activation_Function(cfg.activation)

    def __call__(self, rngs: PRNGKeyArray, x: jax.Array) -> jax.Array:
        cfg = self.config
        # nt x nx x in_dim -> in_dim x nt x nx (channel first representation)
        x = x.transpose((2, 0, 1))
        x = self.fc0(x)
        for i, (speconv, conv) in enumerate(zip(self.sp_convs, self.convs)):
            # hidden_dim x nt x nx
            x1 = speconv(x)
            # hidden_dim x nt x nx
            x2 = conv(x)
            # hidden_dim x nt x n
            x = x1 + x2
            x = self.activation_fn(x)
        # hidden_dim x nt x nx
        x = self.activation_fn(self.fc1(x))
        # out_dim x nt x nx
        x = self.fc2(x)
        if cfg.activate_last_layer:
            x = self.activation_fn(x)
        # out_dim x nt x nx -> nt x nx x out_dim
        x = x.transpose((1, 2, 0))
        return x
