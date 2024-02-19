import jax
import typing

from ..utils.dataclasses import dataclass_wrapper


@dataclass_wrapper
class SparseGrid:
    temporal: jax.Array
    spatial: typing.Iterable[jax.Array]

    def __init__(self, temporal, *spatial) -> 'SparseGrid':
        self.temporal = temporal
        self.spatial = spatial

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def ndim(self) -> int:
        return 1 + len(self.spatial)

    @property
    def shape(self) -> typing.Tuple[int]:
        return (self.temporal.shape[0], *self.spatial[0].shape, self.ndim)


def _geometry_sparse_grid_flatten(grid: SparseGrid) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('temporal'), grid.temporal),
        (jax.tree_util.GetAttrKey('spatial'), grid.spatial)
    ]
    return flat_contents, dict()


def _geometry_sparse_grid_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> SparseGrid:
    return SparseGrid(*data)


jax.tree_util.register_pytree_with_keys(
    SparseGrid, _geometry_sparse_grid_flatten, _geometry_sparse_grid_unflatten)


@dataclass_wrapper
class DenseGrid:
    grid: jax.Array

    def __init__(self, grid: jax.Array) -> 'DenseGrid':
        self.grid = grid

    @property
    def is_sparse(self) -> bool:
        return False

    @property
    def ndim(self) -> int:
        return self.grid.shape[-1]

    @property
    def shape(self) -> typing.Tuple[int]:
        return self.grid.shape


def _geometry_dense_grid_flatten(grid: DenseGrid) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('grid'), grid.grid)
    ]
    return flat_contents, dict()


def _geometry_dense_grid_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> DenseGrid:
    return DenseGrid(*data)


jax.tree_util.register_pytree_with_keys(
    DenseGrid, _geometry_dense_grid_flatten, _geometry_dense_grid_unflatten)


Grid = typing.Union[SparseGrid, DenseGrid]
