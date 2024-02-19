import typing
import yaml
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    # Required
    name: str
    batch_size: int
    data_root: str

    # Time interval
    t: float = None

    # Grid Configuration
    grid_shape: typing.Tuple[int] = None
    trange: typing.Tuple[float, float] = (None, None)
    xrange: typing.Tuple[float, float] = (None, None)
    yrange: typing.Tuple[float, float] = (None, None)

    # IC Generation
    ic_generator: str = None
    ic_generator_kwargs: typing.Dict[str, typing.Any] = None

    # Time Based Options
    n_history: int = None
    n_future: int = None

    # Dataloading options
    prefetch: int = 4


def load_dataset_config(path: str):
    with open(path, 'r') as fp:
        config_dict = yaml.safe_load(fp)
    return DatasetConfig(**config_dict)


def attach_dataset_size(dataset: typing.Any, size: int):
    setattr(dataset, '_n_batches', size)
    return dataset
