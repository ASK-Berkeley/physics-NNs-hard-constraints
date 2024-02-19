import typing
from abc import ABC, abstractmethod
import tensorflow as tf
import jax
import jax.numpy as jnp
import equinox as eqx

from ..utils import dataclass_wrapper
from ..geometry import Function

#### API Declaration for all PDEs. Underscore indicates private functions ####


class AbstractPDE(eqx.Module):
    name: str = 'AbstractPDE'

    def _tf_schema(self, variable_names: typing.List[str]):
        '''
        Not actually sure if you need this. This seems to work in a notebook?
        Ex:
        ```out = next(iter(raw_dset))
        print(type(out))
        parsed = tf.train.Example.FromString(out.numpy())
        print(type(parsed))
        tensors = {}
        for k, v in parsed.features.feature.items():
            print(k)
            tensors[k] = tf.io.parse_tensor(
                v.bytes_list.value[0], out_type=tf.float64)
            print(tensors[k].shape)
        ```'''
        features = {}
        for var in variable_names:
            features[var] = tf.io.FixedLenFeature(
                [], dtype=tf.string)
        return features

    def reweight_matrix(self, weight: jax.Array, matrix: jax.Array) -> jax.Array:
        return (matrix @ weight)[..., None]

    @abstractmethod
    def compute_boundary_conditions(self, terms: typing.Dict[str, jax.Array]):
        pass
    
    @abstractmethod
    def compute_boundary_condition_terms(self, u: Function, pde_params: Function) -> typing.Dict[str, jax.Array]:
        pass

    @abstractmethod
    def process_input(self, input_dict: typing.Dict[str, tf.Tensor]):
        """
        Converts the deserialized tf record dictionary to model input as a preprocessing step.
        Returns a tuple of (pde_params: Tensor, pde_grid: Mesh, target: DenseGrid)
        """
        pass

    @abstractmethod
    def compute_pde_residual_terms(self, u: Function, pde_params: Function, ic: bool = False) -> typing.Dict[str, jax.Array]:
        pass

    @abstractmethod
    def compute_pde_residual(self, terms: typing.Optional[typing.Dict[str, jax.Array]]) -> typing.Tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def vector_objective_function(self, weight: jax.Array, *args):
        pass

    def scalar_objective_function(self, *args) -> jax.Array:
        return jnp.linalg.norm(self.vector_objective_function(*args))
