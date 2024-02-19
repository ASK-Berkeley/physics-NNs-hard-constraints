from typing import List
import typing
import wandb
from dataclasses import asdict
from os.path import join
from os import makedirs
import jax
import jax.numpy as jnp
import optimistix

from ..pdes import abstract_pde
from .train_utils import TrainingConfig
from ..models.model_utils import ModelConfig
from ..datasets.data_utils import DatasetConfig


class ConfigPrefixes:
    optimizer: str = 'optimizer_'
    model: str = 'model_'
    dataset: str = 'dataset_'
    metrics_tracked: str = 'metrics_tracked'
    wandb_entity: str = 'entity'


class MetricsLogger:
    _metrics: dict
    _log_count: dict
    epoch: int
    has_uncommitted: bool
    sync_ckpts: bool

    def __init__(self, project: str,
                 pde: abstract_pde,
                 keys: List[str],
                 model_args: ModelConfig,
                 train_args: TrainingConfig,
                 data_config: DatasetConfig,
                 sync_ckpts: bool = True,
                 wandb_init: bool = True,
                 wandb_config: typing.Dict = dict(), **kwargs):
        self._metrics = dict()
        self._log_count = dict()
        self.sync_ckpts = sync_ckpts
        self.epoch = 0
        self.has_uncommitted = False
        for k in keys:
            self._metrics[k] = 0.
            self._log_count[k] = 0
        config = {
            'model': asdict(model_args),
            'training': asdict(train_args),
            'dataset': asdict(data_config),
            'pde': pde.name
        }

        config['pde'] = pde.name
        if wandb_init:
            wandb.init(project=project,
                       config=config, **wandb_config)
            # Make checkpoints dirs
        makedirs(join(wandb.run.dir, 'checkpoints'), exist_ok=True)
        makedirs(join(wandb.run.dir, 'best_model'), exist_ok=True)
        self.checkpoint_commit(commit_best=True)

    @property
    def log_dir(self):
        return wandb.run.dir

    def __getitem__(self, k):
        if k not in self._metrics.keys():
            self._metrics[k] = 0.
            self._log_count[k] = 0
        return self._metrics[k]

    def add(self, k, a):
        self.has_uncommitted = True
        if k not in self._metrics.keys():
            self._metrics[k] = 0.
            self._log_count[k] = 0
        self._metrics[k] += a
        self._log_count[k] += 1

    def next_epoch(self):
        self.epoch += 1

    def average(self):
        for k in self._metrics.keys():
            if self._log_count[k] != 0:
                self._metrics[k] = self._metrics[k] / self._log_count[k]

    def reset(self):
        for k in self._metrics.keys():
            self._metrics[k] = 0.
            self._log_count[k] = 0
        self.has_uncommitted = False

    def commit(self, step=None, ignore_zeros: bool = False):
        """
        Returns the metrics dict
        """
        to_log = {k: v for k, v in self._metrics.items()}
        if ignore_zeros:
            keys = list(to_log.keys())
            for k in keys:
                if to_log[k] == 0:
                    del to_log[k]
        if step == None:
            step = wandb.run.step
        wandb.log(to_log, step=step, commit=True)
        self.reset()
        return to_log
    
    def histogram(self, key, data):
        wandb.log({key: wandb.Histogram(data)}, commit=False)

    def save_ckpt(self, ckpt_file):
        artifact = wandb.Artifact(name=f'{wandb.run.id}_model.ckpt',
                                  type='model',
                                  description='Model checkpoint')
        artifact.add_reference('file:' + str(ckpt_file))
        wandb.run.log_artifact(artifact, aliases=['latest_model'])

    def checkpoint_commit(self, commit_best=False):
        pass

    def end(self):
        wandb.finish()

    def log_solver_info(self, solver_statuses: jax.Array, solver_iters: jax.Array, solver_weights: jax.Array):
        if solver_statuses is None or solver_iters is None or solver_weights is None:
            return
        # Both are n_batch x n_solves
        # _log_solver_status(solver_statuses=solver_statuses)
        _log_solver_weights(solver_weights=solver_weights)
        _log_solver_iterations(solver_iters=solver_iters)


def _log_solver_weights(solver_weights: jax.Array):
    if solver_weights.ndim == 1:
        # Single solver
        solver_weights = jnp.expand_dims(solver_weights, axis=0)
    # Case with multiple experts
    for i, weights in enumerate(solver_weights):
        wandb.log({f'solver/{i}/weights': wandb.Histogram(weights)}, commit=False)


def _log_solver_iterations(solver_iters: jax.Array):
    iters = solver_iters
    if iters.ndim == 0:
        # Case no experts and just a batch dim
        iters = jnp.expand_dims(iters, axis=0)
    wandb.log({f'solver/{i}/iterations': iteration for i,
              iteration in enumerate(iters)}, commit=False)


def _log_solver_status(solver_statuses: jax.Array):
    data = [
        ['successful', jnp.where(solver_statuses ==
                                 optimistix.RESULTS.successful, 1, 0).sum()],
        ['max_iter_reached', jnp.where(
            solver_statuses == optimistix.RESULTS.max_steps_reached, 1, 0).sum()],
        ['nonlinear_max_iter_reached', jnp.where(
            solver_statuses == optimistix.RESULTS.nonlinear_max_steps_reached, 1, 0).sum()],
        ['nonlinear_divergence', jnp.where(
            solver_statuses == optimistix.RESULTS.nonlinear_divergence, 1, 0).sum()],
        ['singular', jnp.where(
            solver_statuses == optimistix.RESULTS.singular, 1, 0).sum()],
        ['breakdown', jnp.where(
            solver_statuses == optimistix.RESULTS.breakdown, 1, 0).sum()],
        ['stagnation', jnp.where(
            solver_statuses == optimistix.RESULTS.stagnation, 1, 0).sum()],
    ]
    total = jnp.where(solver_statuses ==
                      optimistix.RESULTS.successful, 1, 0).ravel().shape[0]
    num_accounted = 0
    for _, count in data:
        num_accounted += count
    data.append(['unaccounted', total - num_accounted])
    wandb.log({f'solver/{k}': v for k, v in data}, commit=False)
