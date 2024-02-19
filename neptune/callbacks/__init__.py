import wandb

import jax
import jax.numpy as jnp

from . import io, visualize, callback_utils
from ..types import CallbackPayload


def visualization_callback(payload: CallbackPayload):
    visualize.log_heatmaps(payload)

def error_callback(payload: CallbackPayload):

    pred = payload.predicted_solution.image
    true = payload.pde_sol.image

    def _error_callback(pred, true):
        nt, nx, ny, _ = pred.shape

        errs = jnp.linalg.norm((pred - true).reshape(nt, -1), axis=1)
        return errs / jnp.linalg.norm((true).reshape(nt, -1), axis=1)

    errs = jax.vmap(_error_callback)(pred, true)
    best = jnp.argmin(jnp.linalg.norm(errs, axis=1))

    wandb.log({"error(t)/mean": wandb.plot.line(wandb.Table(data=[[x/63*5, y] for x, y in enumerate(jnp.mean(errs, axis=0))], columns=["time", "error"]), "time", "error")}, commit=False)
    wandb.log({"error(t)/best": wandb.plot.line(wandb.Table(data=[[x/63*5, y] for x, y in enumerate(errs[best])], columns=["time", "error"]), "time", "error")}, commit=False)
