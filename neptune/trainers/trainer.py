import jax
import jax.numpy as jnp
import numpy as np
import time
from tqdm import tqdm
import equinox as eqx
from wandb import Histogram
from ..utils import INFO
from .abstract_trainer import AbstractTrainer, callback_dataloader, build_ckpt
from ..geometry import Function, Dimension
from ..types import CallbackPayload, ModelInput, LoggingMetrics
from ..geometry import mesh_utils as mu
from ..models import get_model
from .train_utils import gen_test_step, gen_train_step, gen_compute_loss, map_batch_to_jax
from .checkpointing import save_checkpoint


class Trainer(AbstractTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

    def init_model_and_optimizer(self):
        out = self.pde.process_input(
            self.batch_preprocess(next(iter(self.dataloader.train_numerical_solver_data.unbatch().take(1)))))
        model_input: ModelInput = out[0]
        pde_sol: Function = out[1]

        target_ic: jax.Array = mu.initial_condition(pde_sol).image
        target_lower_bc: jax.Array = mu.left_boundary_condition(
            pde_sol, dim=Dimension.x).image
        target_upper_bc: jax.Array = mu.right_boundary_condition(
            pde_sol, dim=Dimension.x).image

        INFO(f'PDE Params: {model_input.pde_param.shape}')
        INFO(f'Mesh: {model_input.domain.shape}')
        INFO(f'IC: {target_ic.shape}')
        INFO(f'Lower BC: {target_lower_bc.shape}')
        INFO(f'Upper BC: {target_upper_bc.shape}')
        INFO(f'Target: {pde_sol.shape}')

        model_init_key = self.rng_key_manager.next_key()
        self.model = get_model(self.model_cfg,
                               self.pde,
                               rng_key=model_init_key,
                               model_input=model_input)
        self._optimizer_state = self.optimizer.init(eqx.filter(self.model,
                                                               eqx.is_inexact_array))
        num_model_params = sum(jax.tree_util.tree_map(lambda x: np.prod(x.shape)
                                                      if eqx.is_array(x) else 0,
                                                      jax.tree_util.tree_flatten(self.model)[0]))
        INFO(f'Number of Model Parameters: {num_model_params}')

    def init_dataloaders(self):
        INFO(
            f'Number of training batches: {self.dataloader.train_numerical_solver_steps}')
        INFO(
            f'Number of validation batches: {self.dataloader.validation_numerical_solver_steps}')

    def process_metrics(self, metrics: LoggingMetrics, split: str) -> None:
        if split == 'val':
            self.logger.add(f'val/error', metrics.data_loss)
            self.logger.add(f'val/loss/ic', metrics.ic_loss)
            self.logger.add(f'val/loss/pde', metrics.pde_loss)
            self.logger.add(f'val/loss/left_bc', metrics.left_bc_loss)
            self.logger.add(f'val/loss/right_bc', metrics.right_bc_loss)

        else:
            self.logger.add(f'error', metrics.data_loss)

            self.logger.add(f'loss/ic', metrics.ic_loss)
            self.logger.add(f'loss/pde', metrics.pde_loss)
            self.logger.add(f'loss/left_bc', metrics.left_bc_loss)
            self.logger.add(f'loss/right_bc', metrics.right_bc_loss)

            self.logger.add(f'training/loss', metrics.loss)
            self.logger.add(f'training/grad', metrics.grad_norm)
            self.logger.add(f'training/delta', metrics.weight_deltas)
            self.logger.add(f'training/gt_residual', metrics.gt_residual)

            self.logger.log_solver_info(
                metrics.solver_status, metrics.solver_iter, metrics.solver_weight)

            self.logger.commit(step=self.step, ignore_zeros=True)

    def setup_loss(self):
        self._compute_loss = gen_compute_loss(
            ic_loss_weight=self.training_cfg.ic_loss_weight,
            bc_loss_weight=self.training_cfg.bc_loss_weight,
            interior_loss_weight=self.training_cfg.interior_loss_weight,
            pde_loss_weight=self.training_cfg.pde_loss_weight,
            pde=self.pde,
            data_loss_normalize=self.training_cfg.data_loss_normalize,
            icbc_loss_normalize=self.training_cfg.icbc_loss_normalize)

    def compile_model_funcs(self):
        value_and_grad_fn = eqx.filter_value_and_grad(
            self._compute_loss,
            has_aux=True)
        self.train_step = gen_train_step(
            value_and_grad_fn=value_and_grad_fn,
            optimizer=self.optimizer,
            jit=self.jit)
        self.test_step = gen_test_step(
            compute_loss_fn=self._compute_loss,
            jit=self.jit)

    def init_callbacks(self, num_callback_points: int):
        INFO(f'Using {num_callback_points} data points for callbacks.')
        self.train_callback_dataloader = callback_dataloader(self.dataloader.train_numerical_solver_data,
                                                             num_callback_points,
                                                             batch_size=self.dataset_cfg.batch_size)
        self.validation_callback_dataloader = callback_dataloader(self.dataloader.validation_numerical_solver_data,
                                                                  num_callback_points,
                                                                  batch_size=self.dataset_cfg.batch_size)

    def batch_preprocess(self, batch):
        batch = map_batch_to_jax(batch)
        return batch

    def _train(self):
        self.main_pbar = tqdm(self.dataloader.train_numerical_solver_data,
                              total=self.dataloader.train_numerical_solver_data._n_batches,
                              desc=f'Epoch {self.epoch}', disable=not self.training_cfg.use_tqdm)
        num_steps_measured = -1
        for batch in self.main_pbar:
            if self.step % self.log_every_n_steps == 0:
                self.log_step()
                self.main_pbar.unpause()
            batch = self.batch_preprocess(batch)
            loss, metrics, self.model, self._optimizer_state = self.train_step(
                batch,
                self.model,
                self._optimizer_state,
                {'sampler': self.rng_key_manager.get_n_keys(self.dataset_cfg.batch_size)})
            self.main_pbar.set_description((
                f'Loss: {loss:.4f} | '
                f'IC Loss: {metrics.ic_loss:.4f} | '
                f'Data Loss: {metrics.data_loss:.4f} | '
                f'PDE Loss: {metrics.pde_loss:.4f} | '
                f'GT PDE Loss: {metrics.gt_residual:.4f} | '
            ))
            self.process_metrics(metrics, 'train')
            self.step += 1
            num_steps_measured += 1
            if self.training_cfg.max_steps != -1 and self.step >= self.training_cfg.max_steps:
                return

        self.main_pbar.close()

    def _validiation(self):
        self.secondary_pbar = tqdm(self.dataloader.validation_numerical_solver_data,
                                   total=self.dataloader.validation_numerical_solver_data._n_batches,
                                   desc=f'Step {self.step} (Validation)',
                                   leave=False,
                                   disable=not self.training_cfg.use_tqdm)
        losses = []
        for batch in self.secondary_pbar:
            batch = self.batch_preprocess(batch)
            loss, metrics = self.test_step(batch,
                                           self.model,
                                           {'sampler': self.rng_key_manager.get_n_keys(self.dataset_cfg.batch_size)})
            losses.append(metrics.data_loss)
        # Make sure to commit metrics after validation loop
        self.logger.add('val/error', np.mean(losses))
        self.logger.add('val/error_std', np.std(losses))
        self.logger.commit()
        self.secondary_pbar.close()


    def run_callbacks(self, dataloader, split):
        self.secondary_pbar = tqdm(dataloader,
                                   total=(self.num_callback_points //
                                          self.dataset_cfg.batch_size)+1,
                                   desc=f'Step {self.step} ({split[0].upper() + split[1:]} Callbacks)',
                                   leave=False,
                                   disable=not self.training_cfg.use_tqdm)
        payloads = []
        for idx, batch in enumerate(self.secondary_pbar):
            batch = self.batch_preprocess(batch)
            out = jax.vmap(self.pde.process_input)(batch)
            model_input: ModelInput = out[0]
            pde_sol: Function = out[1]
            loss, metrics = self.test_step(batch,
                                           self.model,
                                           {'sampler': self.rng_key_manager.get_n_keys(pde_sol.image.shape[0])})
            callback_payload = CallbackPayload(epoch=self.epoch,
                                               loss=loss,
                                               predicted_solution=metrics.predicted_solution,
                                               pde_sol=pde_sol,
                                               pde_param=model_input.pde_param,
                                               pde=self.pde,
                                               split=split,)
            payloads.append(callback_payload)
        final_payloads = []
        curr_payload: CallbackPayload = None
        for payload in payloads:
            if curr_payload is None:
                curr_payload = payload
            elif curr_payload.pde_sol.image.shape[0] + payload.pde_sol.image.shape[0] < self.num_callback_points:
                curr_payload = curr_payload.merge(payload)
            else:
                final_payloads.append(curr_payload)
                curr_payload = None
        if curr_payload is not None:
            final_payloads.append(curr_payload)

        for payload in final_payloads:
            for callback in self.callbacks:
                callback(payload)
        self.secondary_pbar.close()

    def log_step(self, init: bool = False):
        best_so_far = self._best_val_loss > self.logger['loss/val']
        self._best_val_loss = min(self._best_val_loss, self.logger['loss/val'])
        self._validiation()
        if self.training_cfg.save_checkpoints:
            ckpt_file = save_checkpoint(build_ckpt(self), step=self.step,
                                        best_so_far=best_so_far)
            self.logger.save_ckpt(ckpt_file)
        self.run_callbacks(self.train_callback_dataloader, split='train')
        if self.training_cfg.save_checkpoints:
            self.logger.checkpoint_commit(commit_best=best_so_far)

    def run_epoch(self):
        self._train()
        self.logger.next_epoch()
        self.epoch += 1

    def train(self):
        INFO('Begin Training.')
        current_epoch = self.epoch
        for _ in range(current_epoch, self.training_cfg.num_epochs):
            if self.training_cfg.max_steps == -1 or self.step < self.training_cfg.max_steps:
                self.run_epoch()
        self.cleanup()
        INFO('Done Training!')

    def cleanup(self):
        for fn in self.finish_callbacks:
            fn()
        if self.logger.has_uncommitted:
            self.log_step()
        self.logger.end()
