import random
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf


class RNG_Manager:
    """
    Way to key splits consistent
    """

    def __init__(self, initial_seed: int, seed_libs: bool = True):
        self.key = jax.random.PRNGKey(initial_seed)
        self.splits = 0
        self.initial_seed = initial_seed
        if seed_libs:
            self._seed_libs()

    def next_key(self):
        self.key, subkey = jax.random.split(self.key)
        self.splits += 1
        return subkey

    def get_n_keys(self, n: int):
        keys = []
        for _ in range(n):
            keys.append(self.next_key())
        return jnp.asarray(keys)

    @property
    def split_count(self) -> int:
        return self.splits

    def _seed_libs(self):
        random.seed(self.initial_seed)
        np.random.seed(self.initial_seed)
        tf.random.set_seed(self.initial_seed)
