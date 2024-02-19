import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from typing import List
from os.path import join
import tempfile

DEFAULT_CMAP = 'viridis'
DEFAULT_CENTER_CMAP = 'viridis'


def make_heatmap(fig, ax, image_arr, vmin, vmax, cmap=DEFAULT_CMAP):
    """Plots a single heatmap."""

    im = ax.imshow(image_arr, cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="none",
                   origin='lower')
    plt.axis("off")

    plt.tick_params(
        which="both",
        left=False,
        bottom=False,
        top=False,
        labelleft="off",
        labelbottom="off",
    )
    return ax, im


def transpose_arrs(x):
    return jnp.transpose(x, (0, 2, 1))


def plot_heatmap_grid(image_arrays, vmin, vmax, title="Prediction", figsize=(12, 10), cmap_center=True):
    """Plots image_arrays as a 3x3 grid.

    We supose that image_arrays is shape (9, n, n).

    type: "Prediction" , "True" , "Diff"
    figsize: Size of the overall figure.
    """
    columns = 3
    rows = 3
    if cmap_center:
        cmap = DEFAULT_CENTER_CMAP
    else:
        cmap = DEFAULT_CMAP
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    axes = []
    for index, image_array in enumerate(image_arrays[:9]):
        ax = fig.add_subplot(rows, columns, index + 1)
        ax, im = make_heatmap(fig, ax, image_array, vmin, vmax, cmap=cmap)
        axes.append(ax)

    fig.colorbar(im, ax=axes)

    return fig



def plot_scalar_field(field: jnp.ndarray, ax, vmin=None, vmax=None, cmap=DEFAULT_CMAP):
    im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    return ax


def format_scalar_field_ax(ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('off')
    return ax


def relative_plot_scalar_fields(*fields, cmap=DEFAULT_CMAP):
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields)
    max_val = max(jax.tree_util.tree_map(lambda x: jnp.max(x), fields))
    min_val = max(jax.tree_util.tree_map(lambda x: jnp.min(x), fields))
    vmax = max(abs(max_val), abs(min_val))
    for i, field in enumerate(fields):
        plot_scalar_field(field,
                          format_scalar_field_ax(axes[i]), vmin=-vmax, vmax=vmax, cmap=cmap)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=axes)
    cbar.mappable.set_clim(-vmax, vmax)
    return fig, axes


def absolute_plot_scalar_fields(*fields, cmap=DEFAULT_CMAP):
    n_fields = len(fields)
    fig, axes = plt.subplots(1, n_fields)
    for i, field in enumerate(fields):
        ax = plot_scalar_field(field,
                               format_scalar_field_ax(axes[i]), cmap=cmap)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
        max_val = jnp.max(field)
        min_val = jnp.min(field)
        vmax = max(abs(max_val), abs(min_val))
        cbar.mappable.set_clim(-vmax, vmax)
    return fig, axes


def flatten_scalar_field(field: jnp.ndarray) -> jnp.ndarray:
    field = jnp.squeeze(field)
    if len(field.shape) == 3 and (field.shape[0] == 1 or field.shape[2] == 1):
        return jnp.squeeze(field)
    elif len(field.shape) == 2:
        return field
    else:
        assert False, f'Tried to canonicalize invalid scalar field - shape: {field.shape}'


def plot_truth_vs_predicted_scalar_field(ground_truth: jnp.ndarray, predicted: jnp.ndarray, plot_diffs=True, relative=False, cmap=DEFAULT_CMAP, fig_title: str = 'visualization'):
    ground_truth = flatten_scalar_field(ground_truth)
    predicted = flatten_scalar_field(predicted)
    fields = [ground_truth, predicted]
    if plot_diffs:
        fields.append(ground_truth - predicted)
    fig, axes = relative_plot_scalar_fields(
        *fields, cmap=cmap) if relative else absolute_plot_scalar_fields(*fields, cmap=cmap)
    axes[0].set_title('Target')
    axes[1].set_title('Predicted')
    if plot_diffs:
        axes[2].set_title('Error')
    fig.suptitle(fig_title)
    return fig, axes


# GIF creation

def create_frame(min_val, max_val, field, cmap='RdBu', title='animation'):
    fig, ax = plt.subplots(1, 1)
    plot_scalar_field(field,
                      format_scalar_field_ax(ax), norm=plt.Normalize(min_val, max_val), cmap=cmap)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    cbar.mappable.set_clim(min_val, max_val)
    ax.set_title(title)
    return fig, ax


def save_frames(directory, frames):
    for i, frame in enumerate(frames):
        fig, ax = frame
        fig.savefig(join(directory, f'frame_{i}.png'))


def gif_from_images(directory, n_frames, output_file, duration=0.1):
    frames = [imageio.imread(
        join(directory, f'frame_{i}.png')) for i in range(n_frames)]
    imageio.mimsave(output_file, frames, fps=6)


def create_gif(fields: List[jnp.ndarray], output_file, title='animation', duration=0.1):
    max_val = max(jax.tree_util.tree_map(lambda x: jnp.max(x), fields))
    min_val = max(jax.tree_util.tree_map(lambda x: jnp.min(x), fields))
    frames = [create_frame(min_val, max_val, field, title=title)
              for field in fields]
    with tempfile.TemporaryDirectory() as temporary_directory:
        save_frames(temporary_directory, frames)
        gif_from_images(temporary_directory, len(frames),
                        output_file=output_file, duration=duration)
