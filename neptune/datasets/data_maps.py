import tensorflow as tf
from os.path import join
from os import listdir
from typing import Callable

from ..constants import DatasetSplit
from ..pdes import abstract_pde
from .data_utils import DatasetConfig


def tf_dataset_generator_wrapper(example: tf.train.Example, generator: Callable, generator_signature, *generator_args, **generator_kwargs):
    """
    Takes example and returns a new tf.data.Dataset after applying generator. 
    """
    return generator(example=example, **generator_kwargs)


def load_tf_records(split: DatasetSplit, data_directory: str):
    """
    Returns a dataset of files - NOT A TF RECORD DATASET!
    """
    split = str(split)
    tf_records = [join(data_directory, split, f) for f in listdir(
        join(data_directory, split)) if f[-len('.tfrecords'):] == '.tfrecords']
    dataset = tf.data.Dataset.list_files(
        join(data_directory, split, '*.tfrecords'))
    return dataset, len(tf_records)


def read_tf_records(dataset: tf.data.Dataset):
    dataset = tf.data.TFRecordDataset(dataset,
                                      num_parallel_reads=tf.data.AUTOTUNE)
    return dataset


def deserialize_tfrecord(example: tf.train.Example):
    '''
    Given a featurized example, parses each tensor field.
    '''
    for k in example.keys():
        # TODO I think there should be an conditional check in case there
        # are other types of serialized objects.
        example[k] = tf.io.parse_tensor(example[k], out_type=tf.float64)
    return example


def parse_tfrecord(example: tf.train.Example, schema):
    return tf.io.parse_example(example, schema)


def dataset_size(dataset: tf.data.Dataset, pde: abstract_pde, cfg: DatasetConfig, n_tf_records: int):
    '''
    TODO Right now returns n_tf_records * 1 assuming there is 1 data point per tf_record file.
    Ideally, should return the number of batches in the dataset.
    '''
    n_trajectories_per_tf_record = 1
    size = (n_tf_records * n_trajectories_per_tf_record) // cfg.batch_size
    if size == 0:
        size = 1
    return size
