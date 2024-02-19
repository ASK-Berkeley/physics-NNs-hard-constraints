import typing
import jax
import jax.numpy as jnp

from .grid import SparseGrid, DenseGrid
from .function import Function
from .utils import Dimension


Geometry_T = typing.Union[Function, SparseGrid, DenseGrid]
Dense_T = typing.Union[DenseGrid, Function]


def initial_condition(o: Geometry_T) -> Geometry_T:
    return left_boundary_condition(o, dim=Dimension.t)


""" Lower / left boundary conditions """


def left_boundary_condition(o: Geometry_T, dim: Dimension) -> Geometry_T:
    if isinstance(o, SparseGrid):
        return _left_boundary_condition_sparse(o, dim)
    elif isinstance(o, DenseGrid):
        return _left_boundary_condition_dense(o, dim)
    elif isinstance(o, Function):
        return _left_boundary_condition_function(o, dim)
    elif isinstance(o, jax.Array):
        return _left_boundary_condition_dense(DenseGrid(o), dim).grid


def _left_boundary_condition_sparse(o: SparseGrid, dim: Dimension) -> SparseGrid:
    temporal = o.temporal
    spatial = o.spatial
    if dim == Dimension.t:
        temporal = temporal[0]
        temporal = jnp.expand_dims(temporal, axis=0)
    else:
        spatial = [spatial[i] if i != dim.value-1 else jnp.expand_dims(
            spatial[dim.value-1][0], axis=0) for i in range(len(spatial))]
    return SparseGrid(temporal, *spatial)


def _left_boundary_condition_dense(o: DenseGrid, dim: Dimension) -> DenseGrid:
    grid = o.grid
    grid = jnp.take(grid, 0, axis=dim.value)
    grid = jnp.expand_dims(grid, axis=dim.value)
    return DenseGrid(grid)


def _left_boundary_condition_function(o: Function, dim: Dimension) -> Function:
    domain = o.domain
    image = o.image
    domain = left_boundary_condition(domain, dim)
    image = jnp.take(image, 0, axis=dim.value)
    image = jnp.expand_dims(image, axis=dim.value)
    return Function(domain, image)


""" Upper / right boundary conditions """


def right_boundary_condition(o: Geometry_T, dim: Dimension) -> Geometry_T:
    if isinstance(o, SparseGrid):
        return _right_boundary_condition_sparse(o, dim)
    elif isinstance(o, DenseGrid):
        return _right_boundary_condition_dense(o, dim)
    elif isinstance(o, Function):
        return _right_boundary_condition_function(o, dim)
    elif isinstance(o, jax.Array):
        return _right_boundary_condition_dense(DenseGrid(o), dim).grid


def _right_boundary_condition_sparse(o: SparseGrid, dim: Dimension) -> SparseGrid:
    temporal = o.temporal
    spatial = o.spatial
    if dim == Dimension.t:
        temporal = temporal[-1]
    else:
        spatial = [spatial[i] if i != dim.value-1 else jnp.expand_dims(
            spatial[dim.value-1][-1], axis=0) for i in range(len(spatial))]
    return SparseGrid(temporal, *spatial)


def _right_boundary_condition_dense(o: DenseGrid, dim: Dimension) -> DenseGrid:
    grid = o.grid
    grid = jnp.take(grid, -1, axis=dim.value)
    grid = jnp.expand_dims(grid, axis=dim.value)
    return DenseGrid(grid)


def _right_boundary_condition_function(o: Function, dim: Dimension) -> Function:
    domain = o.domain
    image = o.image
    domain = right_boundary_condition(domain, dim)
    image = jnp.take(image, -1, axis=dim.value)
    image = jnp.expand_dims(image, axis=dim.value)
    return Function(domain, image)


""" Full boundary conditions """


def boundary_condition(o: Geometry_T, dim: Dimension) -> Geometry_T:
    if isinstance(o, SparseGrid):
        return _boundary_condition_sparse(o, dim)
    elif isinstance(o, DenseGrid):
        return _boundary_condition_dense(o, dim)
    elif isinstance(o, Function):
        return _boundary_condition_function(o, dim)
    elif isinstance(o, jax.Array):
        return _boundary_condition_dense(DenseGrid(o), dim).grid


def _boundary_condition_sparse(o: SparseGrid, dim: Dimension) -> SparseGrid:
    temporal = o.temporal
    spatial = o.spatial
    if dim == Dimension.t:
        temporal = jnp.take(temporal, jnp.array([0, -1]), axis=0)
    else:
        spatial = [spatial[i] if i != dim.value - 1 else jnp.take(
            spatial[dim.value-1], jnp.array([0, -1]), axis=0) for i in range(len(spatial))]
    return SparseGrid(temporal, *spatial)


def _boundary_condition_dense(o: DenseGrid, dim: Dimension) -> DenseGrid:
    grid = o.grid
    grid = jnp.take(grid, jnp.array([0, -1]), axis=dim.value)
    return DenseGrid(grid)


def _boundary_condition_function(o: Function, dim: Dimension) -> Function:
    domain = o.domain
    image = o.image
    domain = boundary_condition(domain, dim)
    image = jnp.take(image, jnp.array([0, -1]), axis=dim.value)
    return Function(domain, image)


""" Full Interior """


def interior(o: typing.Union[Geometry_T, jax.Array]) -> Geometry_T:
    if isinstance(o, SparseGrid):
        return _interior_sparse(o)
    elif isinstance(o, DenseGrid):
        return _interior_dense(o)
    elif isinstance(o, Function):
        return _interior_function(o)
    elif isinstance(o, jax.Array):
        return _interior_dense(DenseGrid(o)).grid
    else:
        return None


def _interior_sparse(o: SparseGrid) -> SparseGrid:
    temporal = o.temporal
    spatial = o.spatial
    temporal = temporal[1:-1]
    spatial = [s[1:-1] for s in spatial]
    return SparseGrid(temporal, *spatial)


def _interior_dense(o: DenseGrid) -> DenseGrid:
    grid = o.grid
    start_idxs = tuple(1 for _ in range(len(grid.shape) - 1)) + (0,)
    end_idxs = tuple(i-1 for i in grid.shape[:-1]) + (grid.shape[-1],)
    grid = jax.lax.slice(grid, start_idxs, end_idxs)
    return DenseGrid(grid)


def _interior_function(o: Function) -> Function:
    image = o.image
    domain = o.domain
    domain = interior(domain)
    start_idxs = tuple(1 for _ in range(len(image.shape) - 1)) + (0,)
    end_idxs = tuple(i-1 for i in image.shape[:-1]) + (image.shape[-1],)
    image = jax.lax.slice(image, start_idxs, end_idxs)
    return Function(domain, image)


""" General Transformations """


def to_dense(o: Geometry_T) -> Dense_T:
    if isinstance(o, SparseGrid):
        return _to_dense_sparse(o)
    elif isinstance(o, DenseGrid):
        return _to_dense_dense(o)
    elif isinstance(o, Function):
        return _to_dense_function(o)
    else:
        return None


def _to_dense_sparse(o: SparseGrid) -> DenseGrid:
    temporal = o.temporal
    spatial = o.spatial
    mesh_arrays = jnp.meshgrid(temporal, *spatial, indexing='ij')
    grid = jnp.stack(mesh_arrays, axis=-1)
    return DenseGrid(grid)


def _to_dense_dense(o: DenseGrid) -> DenseGrid:
    return o


def _to_dense_function(o: Function) -> Function:
    return Function(to_dense(o.grid), o.values)
