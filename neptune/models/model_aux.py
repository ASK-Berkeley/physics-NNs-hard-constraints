import jax
from typing import Dict, Union, Optional
import equinox as eqx


class ModelAuxInfo(eqx.Module):
    evaluation_partials: Optional[Dict[str, jax.Array]] = None
    evaluation_predictions: Optional[jax.Array] = None
    evaluation_mesh: Optional[object] = None
    evaluation_pde_params: Optional[jax.Array] = None
    correction_factor_norm: Optional[Union[jax.Array, float]] = None
    upper_bc_partials: Optional[Dict[str, jax.Array]] = None
    solver_error: Optional[Union[jax.Array, float]] = None
    solver_iter: Optional[Union[jax.Array, int]] = None
