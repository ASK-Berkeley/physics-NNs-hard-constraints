from functools import partial

from ..constants import DatasetSplit
from .data_utils import DatasetConfig, attach_dataset_size
from .data_maps import load_tf_records, parse_tfrecord, deserialize_tfrecord, dataset_size, read_tf_records
from ..pdes import DiffusionSorption1D as pde


def Diffusion_Sorption1d_Dataset(split: DatasetSplit, cfg: DatasetConfig):
    train = split == DatasetSplit.Train
    dataset, num_files = load_tf_records(split, cfg.data_root)
    if train:
        dataset = dataset.shuffle(num_files)
    size = dataset_size(dataset, pde, cfg, num_files)
    dataset = dataset.apply(read_tf_records)
    dataset = dataset.map(partial(parse_tfrecord, schema=pde().tf_schema()))
    dataset = dataset.map(deserialize_tfrecord)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)
    dataset = attach_dataset_size(dataset, size)
    return dataset


def Diffusion_Sorption1d_Dataloaders(cfg: DatasetConfig):
    return Diffusion_Sorption1d_Dataset(DatasetSplit.Train, cfg), Diffusion_Sorption1d_Dataset(DatasetSplit.Test, cfg)
