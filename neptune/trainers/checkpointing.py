from typing import Dict
import equinox as eqx
from os.path import join
import wandb
from os import listdir, remove


def save_checkpoint(checkpoint, step, best_so_far=False):
    eqx.tree_serialise_leaves(join(
        wandb.run.dir, 'checkpoints', f'model_{step}.ckpt'), checkpoint
    )
    ckpt_dir = join(wandb.run.dir, 'checkpoints')
    ckpt_files = [f for f in listdir(ckpt_dir) if f.endswith('.ckpt')]
    keep_n = 3
    ckpt_files = [f.split('_')[-1].split('.')[0] for f in ckpt_files]
    ckpt_files = [int(f) for f in ckpt_files]
    if len(ckpt_files) > keep_n:
        ckpt_files.sort()
        i = ckpt_files[0]
        remove(join(ckpt_dir, f'model_{i}.ckpt'))
    return join(wandb.run.dir, 'checkpoints', f'model_{step}.ckpt')


def load_checkpoint(step, target, best_so_far=False):
    ckpt_dir = wandb.run.dir
    if best_so_far:
        ckpt_dir = join(ckpt_dir, 'best_model')
    else:
        ckpt_dir = join(ckpt_dir, 'checkpoints')
    if step == -1 or best_so_far:
        return None
