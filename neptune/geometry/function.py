import jax
from typing import Tuple, Iterable, Mapping
from dataclasses import dataclass

from .grid import Grid

from ..utils.dataclasses import dataclass_wrapper


@dataclass_wrapper
class Function:
    domain: Grid
    image: jax.Array

    def __init__(self, domain: Grid, image: jax.Array) -> 'Function':
        # nt x nx x ny.... x n_dim
        self.domain = domain
        # nt x nx x ny.... x n_channels. n_channels = 1 for scalar functions
        self.image = image

    @property
    def is_sparse(self) -> bool:
        return self.domain.is_sparse

    @property
    def ndim(self) -> int:
        return self.domain.ndim

    @property
    def shape(self) -> Tuple[int]:
        return self.image.shape


def _geometry_function_flatten(function: Function) -> Tuple[Iterable, Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('domain'), function.domain),
        (jax.tree_util.GetAttrKey('image'), function.image)
    ]
    return flat_contents, dict()


def _geometry_function_unflatten(aux_data: Mapping, data: Iterable) -> Function:
    return Function(*data)


jax.tree_util.register_pytree_with_keys(
    Function, _geometry_function_flatten, _geometry_function_unflatten)
