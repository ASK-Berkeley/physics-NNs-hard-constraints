import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass
from typing import Literal, List, Optional


@dataclass(frozen=True)
class MoEConfig:
    split: Literal['spatialtemporal', 'None'] = 'spatialtemporal'
    # Each index is the number of experts in that dimension
    num_experts: Optional[List[int]] = None



def dimension_partition(array: jax.Array, experts: Tuple[int]):
    """(nt, nx, ny, d) -> (N, nt, nx, ny, d)
    """

    array = array[jnp.newaxis]

    for i, split in enumerate(experts, start=1):

        array = jnp.split(array, split, axis=i)
        array = jnp.concatenate(array, axis=0)

    return array


def dimension_unpartition(array: jax.Array, experts: Tuple[int]):
    """(N, nt, nx, ny, d) -> (nt, nx, ny, d)
    """

    for i, split in reversed(list(enumerate(experts, start=1))):

        array = jnp.split(array, split, axis=0)
        array = jnp.concatenate(array, axis=i)

    return array.squeeze(0)