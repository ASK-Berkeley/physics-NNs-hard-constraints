from functools import partial
import tensorflow as tf

from ..constants import DatasetSplit
from .data_utils import DatasetConfig, attach_dataset_size
from .data_maps import load_tf_records, parse_tfrecord, deserialize_tfrecord, dataset_size, read_tf_records, tf_dataset_generator_wrapper
from ..pdes import NavierStokes2D as pde


def Navier_Stokes_2D_Dataset(split: DatasetSplit, cfg: DatasetConfig):
    assert isinstance(split, DatasetSplit)
    train = split == DatasetSplit.Train
    dataset, num_files = load_tf_records(split, cfg.data_root)
    size = dataset_size(dataset, pde, cfg, num_files)
    if train:
        dataset = dataset.shuffle(num_files)
    dataset = dataset.apply(read_tf_records)
    dataset = dataset.map(partial(parse_tfrecord, schema=pde().tf_schema()))
    dataset = dataset.map(deserialize_tfrecord)

    def sample_t(data):
        """
            Clip each record such that the time interval is [0, cfg.t]
            Keep the whole data record if `cfg.t` is not specified
        """

        T = float(data["t"][-1])
        t = cfg.t or T

        N = float(len(data["t"]))
        nt = int(t * (N - 1) / T)

        data["t"] = data["t"][:nt]
        data["u"] = data["u"][:nt]

        return data

    dataset = dataset.map(sample_t)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=train)
    dataset = attach_dataset_size(dataset, size)
    return dataset


def Navier_Stokes_2D_Dataloaders(cfg: DatasetConfig):
    return Navier_Stokes_2D_Dataset(DatasetSplit.Train, cfg), Navier_Stokes_2D_Dataset(DatasetSplit.Validation, cfg)
