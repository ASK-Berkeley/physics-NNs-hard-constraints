from enum import Enum
import logging
from jax.config import config
import tensorflow as tf
import os


def _process_env_var(env_var, default):
    var = os.getenv(env_var, default)
    if not isinstance(var, bool):
        if var.lower() == 'true':
            return True
        else:
            return False
    return var


def setup():
    PRINT_DEBUG_MSGS = _process_env_var('NEPTUNE_DEBUG', False)
    CHECK_NAN = _process_env_var('CHECK_NAN', False)
    JIT = _process_env_var('JIT', True)
    TF_GPUS = []

    '''Setup of the repo configuration'''
    config.update('jax_threefry_partitionable', False)
    config.update("jax_debug_nans", CHECK_NAN)
    config.update('jax_disable_jit', not JIT)

    if not PRINT_DEBUG_MSGS:
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

    import jax.numpy as jnp  # Workaround to make sure jnp is imported after config


class DatasetSplit(Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'
    All = 'All'
