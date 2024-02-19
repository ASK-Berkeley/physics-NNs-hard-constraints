import jax
import jax.numpy as jnp
import tensorflow as tf
import typing
from functools import partial
from .functions import gaussian_rf2d
from .data_utils import DatasetConfig
from .dataset_builder import get_numerical_solver_data
from ..utils import RNG_Manager
from ..types import ModelInput
from ..geometry import Function, SparseGrid


class PDEDataset:
    pde: typing.Any
    cfg: DatasetConfig
    ic_generator: typing.Callable
    rng_manager: RNG_Manager
    validation_numerical_solver_data: tf.data.Dataset
    train_numerical_solver_data: tf.data.Dataset
    domain: SparseGrid

    def __init__(self, cfg: DatasetConfig, pde, rng_generator: RNG_Manager) -> None:
        self.pde = pde
        self.cfg = cfg
        self.rng_manager = rng_generator
        self.setup()

    def setup(self) -> None:
        """
        Sets up the dataset.
        """
        # Setup the IC Generator
        if self.cfg.ic_generator == 'gaussian':
            self.ic_generator = jax.vmap(partial(
                gaussian_rf2d, **self.cfg.ic_generator_kwargs), in_axes=(0,))

        # Setup the Numerical Solver Dataset
        self.train_numerical_solver_data, self.validation_numerical_solver_data = get_numerical_solver_data(
            self.cfg)

    def __call__(self, n_steps: int) -> ModelInput:
        """
        Returns n_steps batches of size batch_size of training data. 
        This data does not have numerical solver data.
        """
        for step in range(n_steps):
            keys = self.rng_manager.get_n_keys(self.cfg.batch_size)
            yield ModelInput(domain=self.domain,
                             pde_param=None,
                             initial_condition=self.ic_generator(keys),
                             boundary_conditions=None)

    def validation_numerical_solver(self) -> typing.Tuple[ModelInput, Function]:
        """
        Returns a batch of size batch_size of numerical solver data.
        """
        for data in self.validation_numerical_solver_data:
            yield data

    def train_numerical_solver(self) -> typing.Tuple[ModelInput, Function]:
        """
        Returns a batch of size batch_size of numerical solver data.
        """
        for data in self.train_numerical_solver_data:
            yield data

    @property
    def train_numerical_solver_steps(self) -> int:
        """
        Returns the number of batches in the train numerical solver dataset.
        """
        return self.train_numerical_solver_data._n_batches

    @property
    def validation_numerical_solver_steps(self) -> int:
        """
        Returns the number of batches in the validation numerical solver dataset.
        """
        return self.validation_numerical_solver_data._n_batches


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def setup_domain(t_bounds, x_bounds, y_bounds, grid_shape):
    """
    Sets up the domain.
    """
    t = jnp.linspace(t_bounds[0], t_bounds[1], grid_shape[0])
    x = jnp.linspace(x_bounds[0], x_bounds[1], grid_shape[1])
    if len(grid_shape) == 2:
        return SparseGrid(t, x)
    elif len(grid_shape) == 3:
        y = jnp.linspace(y_bounds[0], y_bounds[1], grid_shape[2])
        return SparseGrid(t, x, y)
