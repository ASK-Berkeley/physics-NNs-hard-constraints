import jax
import jax.numpy as jnp
import typing
from jaxtyping import PRNGKeyArray
from dataclasses import dataclass

from .grid import Grid, SparseGrid, DenseGrid
from .function import Function
from .mesh_utils import Geometry_T, Dense_T, to_dense

from ..utils.dataclasses import dataclass_wrapper

# Set of masking utilities
# Mask class just wraps the size as mask array


@dataclass_wrapper
class Mask:
    mask: jax.Array
    mask_size: int

    def __init__(self, mask: jax.Array, mask_size: int) -> 'Mask':
        self.mask = mask
        self.mask_size = mask_size

    @property
    def shape(self) -> typing.Tuple[int]:
        return self.mask.shape


def _mask_flatten(mask: Mask) -> typing.Tuple[typing.Iterable, typing.Mapping]:
    flat_contents = [
        (jax.tree_util.GetAttrKey('mask'), mask.mask),
        (jax.tree_util.GetAttrKey('mask_size'), mask.mask_size)
    ]
    return flat_contents, dict()


def _mask_unflatten(aux_data: typing.Mapping, data: typing.Iterable) -> Mask:
    return Mask(*data)


jax.tree_util.register_pytree_with_keys(Mask, _mask_flatten, _mask_unflatten)


def generate_single_mask(o: Geometry_T, mask_size: int, prng_key: PRNGKeyArray) -> Mask:
    # Returns a boolean mask of the same size as the given geometry
    shape = o.shape[:-1]
    mask = jnp.zeros(shape, dtype=jnp.bool_)
    mask = jnp.ravel(mask)
    mask = mask.at[0:mask_size].set(True)
    mask = jax.random.permutation(prng_key, mask, independent=True)
    mask = jnp.reshape(mask, shape)
    return Mask(mask, mask_size)


def generate_masks(o: Geometry_T, mask_sizes: typing.Iterable[int], prng_key: PRNGKeyArray) -> Mask:
    # This is for multiple non-overlapping masks
    n_masks = len(mask_sizes)
    mask = jnp.zeros(o.shape)
    mask = jnp.ravel(mask)
    prev_idx = 0
    idx = 1
    for mask_size in mask_sizes:
        mask = mask.at[prev_idx:prev_idx+mask_size].set(idx)
        idx += 1
        prev_idx += mask_size
    mask = jax.random.permutation(prng_key, mask, independent=True)
    distinct_masks = [jnp.ravel(jnp.zeros_like(
        o.shape), dtype=jnp.bool_) for _ in range(n_masks)]
    for mask_idx in range(n_masks):
        distinct_masks[mask_idx] = distinct_masks.at[mask ==
                                                     mask_idx + 1].set(True)
    return [Mask(mask, mask_size) for mask, mask_size in zip(distinct_masks, mask_sizes)]


def apply_mask(o: Geometry_T, mask: Mask) -> typing.Union[jax.Array, typing.Tuple[jax.Array, jax.Array]]:
    if isinstance(o, SparseGrid):
        return _apply_mask_sparse(o, mask)
    elif isinstance(o, DenseGrid):
        return _apply_mask_dense(o, mask)
    elif isinstance(o, Function):
        return _apply_mask_function(o, mask)
    elif isinstance(o, jax.Array):
        return _apply_mask_dense(DenseGrid(o), mask)
    else:
        return None


def _apply_mask_sparse(o: SparseGrid, mask: Mask) -> jax.Array:
    return _apply_mask_dense(to_dense(o), mask)


def _apply_mask_dense(o: DenseGrid, mask: Mask) -> jax.Array:
    grid = o.grid
    grid = jnp.reshape(grid, (-1, grid.shape[-1]))
    idx = jnp.nonzero(mask.mask.ravel(), size=mask.mask_size)[0]
    return grid[idx]


def _apply_mask_function(o: Function, mask: Mask) -> typing.Tuple[jax.Array, jax.Array]:
    domain = _apply_mask_dense(to_dense(o.domain), mask)
    idx = jnp.nonzero(mask.mask.ravel(), size=mask.mask_size)[0]
    image = jnp.reshape(o.image, (-1, o.image.shape[-1]))
    image = image[idx]
    return domain, image
