from argparse import ArgumentParser

from neptune.trainers.train_utils import load_train_config
from neptune.trainers import Trainer
from neptune.trainers.train_utils import TrainingConfig
from neptune.models import get_model_config
from neptune.datasets import load_dataset_config
from neptune.trainers.metrics import MetricsLogger
from neptune.callbacks import visualization_callback
from neptune.pdes import DiffusionSorption1D, NavierStokes2D

# force square plots
import matplotlib as mpl
mpl.rcParams["image.aspect"] = 'auto'


def main():
    parser = ArgumentParser('Main Training Script')
    parser.add_argument('-m', '--model_config', type=str,
                        help='path to model config', default='')
    parser.add_argument('-d', '--dataset_config', type=str,
                        help='path to dataset config', default='')
    parser.add_argument('-t', '--training_config', type=str,
                        help='Training Config', default='')
    parser.add_argument('-w', '--wandb_description', type=str, default=None)
    parser.add_argument('-p', '--project', type=str, default='MoE')
    args = parser.parse_args()

    cb = [visualization_callback,]
    end_cb = []

    model_cfg = get_model_config(args.model_config)
    dataset_cfg = load_dataset_config(args.dataset_config)
    training_cfg: TrainingConfig = load_train_config(args.training_config)
    wandb_description = args.wandb_description
    if 'navier-stokes' in args.dataset_config:
        pde = NavierStokes2D()
    else:
        pde = DiffusionSorption1D()
    logger = MetricsLogger(project=args.project,
                           pde=pde,
                           keys=[],
                           model_args=model_cfg,
                           train_args=training_cfg,
                           data_config=dataset_cfg,
                           notes=wandb_description,)
    trainer = Trainer(model_cfg,
                      dataset_cfg,
                      training_cfg,
                      pde=pde,
                      logger=logger,
                      callbacks=cb,
                      finish_callbacks=end_cb)
    trainer.train()


if __name__ == '__main__':
    main()
