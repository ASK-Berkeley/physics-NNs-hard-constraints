from abc import ABC, abstractmethod
from typing import Any, List, Dict, Callable
import optax
import tensorflow as tf
import equinox as eqx
import jax.numpy as jnp

from ..datasets import PDEDataset
from ..types import LoggingMetrics
from ..utils import RNG_Manager
from ..models.model_utils import ModelConfig
from ..pdes import AbstractPDE
from ..types import Callback
from ..datasets.data_utils import DatasetConfig
from .train_utils import TrainingConfig
from .metrics import MetricsLogger
from .optimizers import construct_optimizer
from .checkpointing import load_checkpoint


class AbstractTrainer(ABC):
    # Configs
    model_cfg: ModelConfig
    dataset_cfg: DatasetConfig
    training_cfg: TrainingConfig
    pde: AbstractPDE

    # Trainer State
    rng_key_manager: RNG_Manager
    _compute_loss: Any
    epoch: int
    step: int

    # Model and Optimizer
    optimizer: optax.GradientTransformation
    model: eqx.Module
    _optimizer_state: Any

    # Generated Functions
    train_step: Callable
    test_step: Callable

    # Logging
    logger: MetricsLogger
    _best_val_loss: float
    log_every_n_steps: int
    skipped_logging_keys: List[str]

    # Dataloaders
    dataloader: PDEDataset
    num_callback_points: int

    # Callbacks
    callbacks: List[Callback]
    finish_callbacks: List[Callback]
    callback_batch_size: int

    # Jax performance params
    jit: bool = True

    # Displays
    main_pbar: Any
    secondary_pbar: Any

    @abstractmethod
    def __init__(self, model_cfg: ModelConfig,
                 dataset_cfg: DatasetConfig,
                 training_cfg: TrainingConfig,
                 pde: AbstractPDE,
                 logger: MetricsLogger = None,
                 callbacks: List[Callback] = [],
                 finish_callbacks: List = []) -> None:
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg
        self.jit = training_cfg.jit
        self.pde = pde
        self.rng_key_manager = RNG_Manager(self.training_cfg.seed)
        self.train_step = None
        self._compute_loss = None
        self.test_step = None
        self.epoch = 0
        self.logger = logger
        self.callbacks = callbacks
        self.finish_callbacks = finish_callbacks
        self.num_callback_points = training_cfg.num_callback_points
        self._best_val_loss = float('inf')
        self.log_every_n_steps = training_cfg.log_every_n_steps
        self.skipped_logging_keys = ['model_output']
        self.step = 0
        self.callback_batch_size = 6
        self.dataloader = PDEDataset(dataset_cfg, pde, self.rng_key_manager)

    def setup(self):
        """
        Helper to make it easier to rerun setup functions
        """
        self.optimizer = construct_optimizer(
            self.training_cfg.optimizer, self.training_cfg.num_epochs)
        self.init_dataloaders()
        self.init_model_and_optimizer()
        self.setup_loss()
        self.compile_model_funcs()
        self.init_callbacks(self.num_callback_points)

    @abstractmethod
    def init_model_and_optimizer(self) -> None:
        """
        Initialize the model and optimizer parameters.
        """
        pass

    @abstractmethod
    def init_dataloaders(self) -> None:
        """
        Initialize the dataloaders and the number of batches per epoch.
        """
        pass

    @abstractmethod
    def setup_loss(self) -> None:
        """
        Setup the loss function.
        """
        pass

    @abstractmethod
    def compile_model_funcs(self) -> None:
        """
        Compile the forward and backward passes of the model.
        """
        pass

    @abstractmethod
    def init_callbacks(self, num_callback_points: int) -> None:
        """
        Initialize the callback dataloaders.
        Setup two loaders with just the correct amount of data points.
        """
        pass

    @abstractmethod
    def batch_preprocess(self, batch: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """
        Function to preprocess a batch of data before passing to compiled forward or backprop steps.
        """
        pass

    @abstractmethod
    def process_metrics(self, metrics: LoggingMetrics, split: str) -> None:
        """
        Process the auxiliary metrics from the model.
        """
        pass

    @abstractmethod
    def _train(self) -> None:
        """
        Training loop for one iteration over the train dataloader.
        """
        pass

    @abstractmethod
    def _validiation(self) -> None:
        """
        Inference loop for one iteration over the validation dataloader.
        """
        pass

    @abstractmethod
    def run_callbacks(self, dataloader: tf.data.Dataset, split: str) -> None:
        """
        Given a dataloader and a split, run the callbacks.
        """
        pass

    @abstractmethod
    def log_step(self, init: bool = False) -> None:
        """
        Log the accumation of metrics over the last self.log_every_n_steps.
        Runs validation and callbacks.
        init: If true, log the initial state of the model.
        """
        pass

    @abstractmethod
    def run_epoch(self) -> None:
        """
        Run one epoch. Performs the training and validation loops. Also runs the callbacks.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Public training function. Runs the training loop for the specified number of epochs.
        Performs self::cleanup when finished.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup function to be run after training is finished.
        """
        pass


"""
Collection of utility functions for the trainer.
"""


def callback_dataloader(dataset: tf.data.Dataset, num_callback_points: int, batch_size: int) -> tf.data.Dataset:
    """
    Generates a dataloader for the callbacks.
    """
    # This step removes the dependence on the original dataset - forces an evaluation
    callback_dataset = list(dataset.unbatch().take(
        num_callback_points).batch(batch_size, drop_remainder=False))
    return callback_dataset


"""
Utility functions for checkpointing.
"""


def build_ckpt(trainer: AbstractTrainer):
    return {
        'model_params': trainer.model,
        'opt_state': trainer._optimizer_state,
        'epoch': trainer.epoch,
        'rng_splits': trainer.rng_key_manager.split_count,
    }


def restore_trainer_state(trainer: AbstractTrainer, ckpt_file):
    restored_state = eqx.tree_deserialise_leaves(ckpt_file,
                                                 build_ckpt(trainer))
    step = int(ckpt_file.split('_')[-1].split('.')[0])
    trainer.step = step + 1
    trainer.model = restored_state['model_params']
    trainer._optimizer_state = restored_state['opt_state']
    trainer.epoch = restored_state['epoch'] + 1
    trainer.logger.epoch = trainer.epoch
    rng_splits = restored_state['rng_splits']
    while trainer.rng_key_manager.split_count < rng_splits:
        trainer.rng_key_manager.next_key()
    return trainer


def restore_trainer_last_step(trainer: AbstractTrainer):
    restore_trainer_state(trainer, -1)


def restore_trainer_best_step(trainer: AbstractTrainer):
    restore_trainer_state(trainer, None)
