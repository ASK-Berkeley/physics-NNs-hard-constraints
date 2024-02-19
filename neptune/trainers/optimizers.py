import optax
from typing import Dict
from .train_utils import OptimizerConfig


def construct_optimizer(config: OptimizerConfig, num_epochs: int):
    return _construct_optimizer(config, num_epochs)


def _construct_optimizer(config: OptimizerConfig, num_epochs: int):
    if config.name == 'adam':
        if config.scheduler == 'ns': # Navier Stokes scheduler
            return construct_adam_ns(learning_rate=config.learning_rate,
                                     num_epochs=num_epochs)
        elif config.scheduler == 'ds': # Diffusion Sorption scheduler
            return construct_adam_ds(learning_rate=config.learning_rate,
                                     num_epochs=num_epochs)
        return construct_adam(learning_rate=config.learning_rate,
                              num_epochs=num_epochs)
    else:
        raise NotImplementedError()


def construct_adam_ds(learning_rate: float, num_epochs: int):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(optax.exponential_decay(init_value=learning_rate,
                                                    end_value=1e-4,
                                                    transition_steps=800,
                                                    staircase=True,
                                                    decay_rate=0.5)),
        optax.scale(-1.0)
    )
    return optimizer

def construct_adam_ns(learning_rate: float, num_epochs: int):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(optax.exponential_decay(init_value=learning_rate, transition_steps=1,
                                                        decay_rate=1e-3 ** (1e-3 / num_epochs))),
        optax.scale(-1.0)
    )
    return optimizer


def construct_adam(learning_rate: float, num_epochs: int):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(optax.exponential_decay(init_value=learning_rate, transition_steps=1, decay_rate=1e-3 ** (1e-3 / num_epochs))),
        optax.scale(-1.0)
    )
    return optimizer


def construct_scheduler(**kwargs):
    assert 'name' in kwargs.keys(), 'Could not find scheduler name!'
    if kwargs['name'] == 'warmup_cosine_decay_schedule':
        del kwargs['name']
        return optax.warmup_cosine_decay_schedule(**kwargs)
    else:
        raise NotImplementedError()
