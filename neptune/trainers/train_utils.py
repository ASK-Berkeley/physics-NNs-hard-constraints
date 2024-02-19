from dataclasses import dataclass
from typing import Mapping
import jax
import jax.numpy as jnp
import functools
import yaml
from jax.debug import print as jprint
import equinox as eqx

from ..pdes import DiffusionSorption1D
from ..models.model_aux import ModelAuxInfo
from ..losses import MSE, PDELoss
from ..types import ModelInput, ModelOutput, LoggingMetrics
from ..geometry import Grid, Function, Mask, Dimension
from ..geometry import mesh_utils as mu


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    scheduler: str = 'ds'


@dataclass
class TrainingConfig:
    num_epochs: int
    optimizer: OptimizerConfig
    seed: int
    save_checkpoints: bool = True
    jit: bool = True
    log_every_n_steps: int = 25
    num_callback_points: int = 16
    ic_loss_weight: float = 1.
    bc_loss_weight: float = 0.
    interior_loss_weight: float = 0.
    pde_loss_weight: float = 1.
    data_loss_normalize: bool = True
    icbc_loss_normalize: bool = True
    use_tqdm: bool = True
    max_steps: int = -1


def load_train_config(path: str):
    with open(path, 'r') as fp:
        config_dict = yaml.safe_load(fp)
    config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
    return TrainingConfig(**config_dict)


def best_validation_loss(metrics: Mapping[str, float]) -> float:
    return metrics['loss/val']


def global_gradient_norm(grad):
    return jax.tree_util.tree_reduce(
        lambda norm, x: norm + x,
        jax.tree_util.tree_leaves(
            jax.tree_map(
                lambda x: jnp.linalg.norm(x) ** 2, grad)
        ),
        initializer=0.0
    )


def gradient_conjugate(grads):
    return jax.tree_map(lambda x: jnp.conj(x), grads)


def add_loss_dict(aux_dict, losses, prefix):
    for k, v in losses.items():
        aux_dict[prefix + k] = v
    return aux_dict

# Helper jit functions


@jax.vmap
def rel_l2_loss_f(x, y):
    return jnp.linalg.norm(x.ravel() - y.ravel()) / jnp.linalg.norm(y.ravel())


def gen_compute_loss(ic_loss_weight=1.,
                     bc_loss_weight=1.,
                     interior_loss_weight=1.,
                     pde_loss_weight=1.,
                     pde=None,
                     data_loss_normalize=True,
                     icbc_loss_normalize=True):
    data_loss_fn = MSE(normalize=data_loss_normalize)
    icbc_loss_fn = MSE(normalize=icbc_loss_normalize)
    pde_loss_fn = PDELoss(pde=pde)

    def _compute_loss(model, batch, rngs):
        out = jax.vmap(pde.process_input)(batch)
        model_input: ModelInput = out[0]
        pde_sol: Function = out[1]
        model_out: ModelOutput = model(rngs, model_input, pde_sol)
        predicted_solution: Function = model_out.solution
        ### IC LOSS ###
        predicted_ic = predicted_solution.image[:, 0]
        gt_ic = pde_sol.image[:, 0]
        batch_size = predicted_ic.shape[0]
        ic_loss = icbc_loss_fn(predicted_ic.reshape((batch_size, -1)),
                               gt_ic.reshape((batch_size, -1)))
        ### BC LOSS ###
        if bc_loss_weight > 0 and pde.name == 'Diffusion-Sorption1D':
            D = 5e-4
            # b x t x space
            left_bc_loss = predicted_solution.image[:, :, 0] - 1.
            left_bc_loss = jnp.linalg.norm(
                left_bc_loss.reshape(batch_size, -1), axis=1).mean()
            right_bc_loss = predicted_solution.image[:, :, -1] \
                - pde_sol.image[:, :, -1]
            right_bc_loss = jnp.linalg.norm(
                right_bc_loss.reshape(batch_size, -1), axis=1).mean()
        else:
            left_bc_loss = 0
            right_bc_loss = 0

        ### DATA LOSS ###
        data_loss = data_loss_fn(predicted_solution.image, pde_sol.image)

        ### PDE LOSS ###
        pde_loss = pde_loss_fn(predicted_solution, model_input.pde_param)

        ### Sum losses ###
        loss = 0.
        loss += ic_loss_weight * ic_loss
        loss += bc_loss_weight * (left_bc_loss + right_bc_loss)
        loss += pde_loss_weight * pde_loss
        loss += interior_loss_weight * data_loss

        ### Misc logging ###
        gt_residual = pde_loss_fn(pde_sol, model_input.pde_param)

        solver_iter = None
        solver_weight = None
        solver_status = None

        # Compute raw l2 error for plotting
        raw_l2_loss = rel_l2_loss_f(predicted_solution.image, pde_sol.image)

        if model_out.weight is not None:
            solver_iter = jnp.mean(model_out.solver_iter, axis=0)
            solver_status = model_out.solver_status
            solver_weight = jnp.linalg.norm(model_out.weight, axis=0)

        metrics = LoggingMetrics(
            predicted_solution=predicted_solution,
            loss=loss,
            ic_loss=ic_loss,
            left_bc_loss=left_bc_loss,
            right_bc_loss=right_bc_loss,
            data_loss=data_loss,
            solver_weight=solver_weight,
            solver_status=solver_status,
            solver_iter=solver_iter,
            pde_loss=pde_loss,
            gt_residual=gt_residual,
            raw_l2_loss=raw_l2_loss,
        )
        return loss, metrics
    return _compute_loss


def gen_test_step(compute_loss_fn, jit=True):
    def test_step(batch, model, rngs, value_fn):
        loss, aux = value_fn(model, batch, rngs)
        return loss, aux
    func = functools.partial(test_step, value_fn=compute_loss_fn)
    if jit:
        return eqx.filter_jit(func)
    else:
        return func


def gen_train_step(value_and_grad_fn, optimizer, jit=True):
    # generic train step
    def train_step(batch, model, optimizer_state, rngs, value_and_grad_fn, optimizer):
        (loss, metrics), grads = value_and_grad_fn(
            model, batch, rngs)
        grads = gradient_conjugate(grads)
        grads = jax.tree_map(lambda x: jnp.nan_to_num(x), grads)
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, model)
        model = eqx.apply_updates(model, updates)
        metrics.grad_norm = global_gradient_norm(grads)
        metrics.weight_deltas = global_gradient_norm(updates)
        return loss, metrics, model, optimizer_state
    func = functools.partial(
        train_step, value_and_grad_fn=value_and_grad_fn, optimizer=optimizer)
    if jit:
        return eqx.filter_jit(func)
    else:
        return func


def map_batch_to_jax(batch):
    return jax.tree_map(lambda x: jnp.array(x), batch)
