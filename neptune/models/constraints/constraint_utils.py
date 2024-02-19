import jax
from dataclasses import dataclass
from typing import Literal, Callable, Union


@dataclass(frozen=True)
class ConstraintConfig:
    system: Literal['equalityqp', 'none',
                    'levenbergmarquardt',
                    'gaussnewton', 'lbfgs'] = 'levenbergmarquardt'
    num_sampled_points: int = 200
    tol: float = 1e-4
    atol: float = 1e-4
    rtol: float = 1e-4
    maxiter: int = 100
    refine_regularization: float = 0.
    refine_maxiter: int = 5
    linear_solver: str = "gmres"
    ridge: float = 1e-4
    damping_parameter: float = 1.
    mask_boundary_conditions: bool = False
    use_jaxopt: bool = True
