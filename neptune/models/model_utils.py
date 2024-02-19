import jax
from dataclasses import dataclass

from .constraints import ConstraintConfig
from .MoE import MoEConfig


@dataclass(frozen=True)
class ModelConfig:
    model: str
    constraint: ConstraintConfig = ConstraintConfig(system='none')
    n_basis: int = 1
    moe_config: MoEConfig = MoEConfig(split='none')


def split_prng_key(key, num_keys=2):
    if num_keys == 2:
        return jax.random.split(key)
    else:
        keys = jax.random.split(key, num=num_keys)
        return keys[0], keys[1:]
