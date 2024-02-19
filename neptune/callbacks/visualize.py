from typing import NoReturn
from tempfile import TemporaryDirectory
import jax.numpy as jnp
from os.path import join
import wandb
import io
from ..types import CallbackPayload
from ..utils import visualization as vis
from ..utils.visualization import plot_heatmap_grid
from PIL import Image
import matplotlib.pyplot as plt


def log_heatmaps(
    payload: CallbackPayload,
) -> None:
    """Makes heatmaps for predicted function, the true function and the difference, and logs them on wandb.

    Args:
      predicted_arrs: arrays of predicted values for one batch
      target_arrs: arrays of target values for the same batch
      batch_idx: Number of processed batches since beginning of training.
      postfix: One of ['train', 'test']
      figsize: size of the figure in inches
      pde_params: Give pde_params if this is to be plotted
    """
    figsize = (6, 5)
    split = payload.split
    predicted_arrs = payload.predicted_solution.image
    target_arrs = payload.pde_sol.image
    index = 1
    target = payload.pde_sol
    predicted = payload.predicted_solution.image
    grid = payload.pde_sol.domain

    # TODO: Just a quick way to make sure NS visualization logic is correct
    if payload.pde.name == 'Navier-Stokes2D':
        log_heatmaps_3d(payload)
        return

    dt = (jnp.max(grid.grid[0]) - jnp.min(grid.grid[0])) / grid.shape[0]
    dx = (jnp.max(grid.grid[1]) - jnp.min(grid.grid[1])) / grid.shape[1]
    has_multi_component = len(target.shape) >= 4 and target.shape[-1] > 1

    if predicted_arrs.shape[-1] == 1:
        predicted_arrs = predicted_arrs.squeeze(-1)
    if target_arrs.shape[-1] == 1:
        target_arrs = target_arrs.squeeze(-1)

    if has_multi_component:
        predicted_arrs = predicted_arrs.transpose((0, 2, 1, 3))
        target_arrs = target_arrs.transpose((0, 2, 1, 3))
    else:
        predicted_arrs = predicted_arrs.transpose((0, 2, 1))
        target_arrs = target_arrs.transpose((0, 2, 1))

    diffs = predicted_arrs - target_arrs

    plot_figs_helper(predicted_arrs,
                     vmin=target_arrs.min(),
                     vmax=target_arrs.max(),
                     split=split,
                     index=index,
                     name=f"Predicted",
                     istarget=False,
                     key="normalized")
    plot_figs_helper(predicted_arrs,
                     vmin=predicted_arrs.min(),
                     vmax=predicted_arrs.max(),
                     split=split,
                     index=index,
                     name=f"Predicted",
                     istarget=False,)
    plot_figs_helper(target_arrs,
                     vmin=target_arrs.min(),
                     vmax=target_arrs.max(),
                     split=split,
                     index=index,
                     name=f"Target",
                     istarget=True,)
    max_abs_diff = abs(diffs).max()
    plot_figs_helper(diffs,
                     vmin=-max_abs_diff,
                     vmax=max_abs_diff,
                     split=split,
                     index=index,
                     name=f"Difference",
                     istarget=False,
                     cmap_center=True)


def log_heatmaps_3d(payload: CallbackPayload) -> None:
    figsize = (6, 5)
    split = payload.split
    predicted_arrs = payload.predicted_solution.image
    target_arrs = payload.pde_sol.image
    index = 1
    target = payload.pde_sol
    predicted = payload.predicted_solution
    grid = payload.pde_sol.domain.grid
    # nt, nx, ny, grid_dim
    t_array = grid[0, :, 0, 0, 0]
    target = jnp.squeeze(target.image)
    plot_kf(target,
            is_gt=True,
            split=split,
            center=False,
            t_array=t_array
            )
    predicted = jnp.squeeze(predicted.image)
    plot_kf(predicted,
            is_gt=False,
            split=split,
            center=False,
            t_array=t_array
            )
    vlim = jnp.abs(target).max() * 0.1
    plot_kf(predicted - target,
            is_gt=False,
            split=split,
            center=True,
            t_array=t_array,
            diff=True,
            vmin=-vlim,
            vmax=vlim)


def plot_kf(array, is_gt, split, center, t_array, diff=False, vmin=None, vmax=None, field_name='u'):
    if is_gt:
        target = 'Target'
    else:
        target = 'Predicted'
    PLOT_EVERY_N = 10
    i = 0
    while i < array.shape[1]:
        curr_step = array[:, i]
        vmin = vmin or curr_step.min()
        vmax = vmax or curr_step.max()
        t = t_array[i]
        plot_heatmap_grid(curr_step,
                          vmin,
                          vmax,
                          title=f'{target} - Step {i} - T = {t}' if not diff else f'Differences - Step {i} - T = {t}',
                          figsize=(6, 5),
                          cmap_center=center)
        if not diff:
            wandb.log(
                {
                    f'{split}/{target}/{field_name}/{i}': plt
                },
                commit=False,
            )
        else:
            wandb.log(
                {
                    f'{split}/{field_name}_difference/{i}': plt
                },
                commit=False,
            )
        plt.close()
        plt.clf()
        i = i + PLOT_EVERY_N


def convert_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return Image.open(buf)


def visualize_pde_relative(payload: CallbackPayload) -> NoReturn:
    predicted_field = payload.predicted
    target_field = payload.pde_sol
    fig, _ = vis.plot_truth_vs_predicted_scalar_field(
        ground_truth=target_field, predicted=predicted_field, plot_diffs=True, relative=True)

    return convert_to_pil(fig)


def visualize_pde_absolute(payload: CallbackPayload) -> NoReturn:
    predicted_field = payload.predicted
    target_field = payload.pde_sol
    fig, _ = vis.plot_truth_vs_predicted_scalar_field(
        ground_truth=target_field, predicted=predicted_field, plot_diffs=True, relative=False)
    return convert_to_pil(fig)


class GIFCreator:
    def __init__(self, output_file, title, wandb_category, idx, split, extractor, key):
        self.temp_dir = TemporaryDirectory()
        self.steps = []
        self.output_file = output_file
        self.title = title
        self.wandb_category = wandb_category
        self.idx = idx
        self.split = split
        self.extractor = extractor
        self.key = key

    def save_array(self, payload: CallbackPayload):
        if self.idx in payload.data_indexes and self.split == payload.split:
            _idx = payload.data_indexes.index(self.idx)
            x = self.extractor(payload)[_idx][self.key]
            self.steps.append(payload.epoch)
            jnp.save(join(self.temp_dir.name, f'{payload.epoch}.npy'), x)

    def finish(self):
        arrays = []
        for step in self.steps:
            arrays.append(jnp.load(join(self.temp_dir.name, f'{step}.npy')))

def plot_figs_helper(arr, split, index, vmin=None, vmax=None, figsize=(6, 5), name="",
                     key=None,
                     istarget=False,
                     cmap_center=False):
    vmin = vmin if vmin is not None else arr.min()
    vmax = vmax if vmax is not None else arr.max()
    prediction_or_target_or_none = 'Target' if istarget == True else 'Prediction'
    plot_heatmap_grid(arr, vmin, vmax,
                      title=f"{prediction_or_target_or_none} - {name}", figsize=figsize, cmap_center=cmap_center)
    key = key if key is not None else name
    if istarget is not None:
        wandb.log(
            {
                split + f"/{index}/{prediction_or_target_or_none.lower()}/{key}": plt,
            },
            commit=False,
        )
    else:
        wandb.log(
            {
                split + f"/{index}/{key}": plt,
            },
            commit=False,
        )

    plt.close()
