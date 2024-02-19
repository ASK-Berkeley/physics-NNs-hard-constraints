from neptune.parser import parse_args
from neptune.callbacks import visualization_callback
from neptune.trainers.metrics import MetricsLogger
from neptune.trainers import Trainer
import jax


def main():
    print('NUMBER OF LOCAL CUDA VISIBLE DEVICES', jax.local_device_count())
    model_cfg, training_cfg, dataset_cfg, pde = parse_args()
    cb = [visualization_callback]
    end_cb = []

    logger = MetricsLogger(project='MoE',
                           pde=pde,
                           keys=[],
                           model_args=model_cfg,
                           train_args=training_cfg,
                           data_config=dataset_cfg)
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
