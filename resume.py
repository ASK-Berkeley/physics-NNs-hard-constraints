import argparse
import wandb
import equinox as eqx
from neptune.pdes import DiffusionSorption1D, NavierStokes2D
from neptune.datasets import DatasetConfig
from neptune.models import dict_to_model_config
from neptune.trainers.optimizers import OptimizerConfig
from neptune.trainers.train_utils import TrainingConfig
from neptune.callbacks import visualization_callback
from neptune.trainers.metrics import MetricsLogger
from neptune.trainers import Trainer
from neptune.trainers import restore_trainer_state


def convert_dict_to_tuples(d):
    if isinstance(d, dict):
        return {
            k: convert_dict_to_tuples(v) for k, v in d.items()
        }
    elif isinstance(d, list) and not isinstance(d, str):
        return tuple(convert_dict_to_tuples(v) for v in d)
    else:
        return d


def main():
    parser = argparse.ArgumentParser(description='Resume a training run')
    parser.add_argument('run_id', type=str, help='wandb run id')
    parser.add_argument('project', type=str, help='wandb project name')
    args = parser.parse_args()
    run_id = args.run_id
    project = args.project
    wandb.init(entity='ml-pde', project=project, id=run_id, resume='must')
    cfg = wandb.run.config.as_dict()
    pde_name = cfg['pde']
    if pde_name == 'Navier-Stokes2D':
        pde = NavierStokes2D()
    elif pde_name == 'Diffusion-Sorption1D':
        pde = DiffusionSorption1D()
    else:
        raise NotImplementedError(f'PDE {pde_name} not implemented')
    cfg = convert_dict_to_tuples(cfg)
    training_cfg = cfg['training']
    training_cfg['optimizer'] = OptimizerConfig(**training_cfg['optimizer'])
    training_cfg = TrainingConfig(**training_cfg)
    dataset_cfg = DatasetConfig(**cfg['dataset'])
    model_cfg = dict_to_model_config(cfg['model'])
    cb = [visualization_callback]
    end_cb = []

    logger = MetricsLogger(project=args.project,
                           pde=pde,
                           keys=[],
                           model_args=model_cfg,
                           train_args=training_cfg,
                           data_config=dataset_cfg,
                           wandb_init=False)
    trainer = Trainer(model_cfg,
                      dataset_cfg,
                      training_cfg,
                      pde=pde,
                      logger=logger,
                      callbacks=cb,
                      finish_callbacks=end_cb)

    ckpt_artifact = wandb.run.use_artifact(
        f'{wandb.run.id}_model.ckpt:latest', type='model')
    ckpt_file = ckpt_artifact.file()
    trainer = restore_trainer_state(trainer, ckpt_file)
    trainer.train()


if __name__ == '__main__':
    main()
