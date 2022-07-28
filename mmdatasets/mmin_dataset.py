"""
supervised dataset provider for Multi-model input
"""
from lumo import DatasetBuilder
from .const import get_root
from .datas import pick_datas


def get_train_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split='train')

    ds = (
        DatasetBuilder()
        .add_input('all', samples)
        .add_output('all', 'all')
        .chain()
    )

    return ds


def get_val_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split='val')

    ds = (
        DatasetBuilder()
        .add_input('all', samples)
        .add_output('all', 'all')
        .chain()
    )

    return ds


def get_test_dataset(dataset_name, method='fb'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split='test')

    ds = (
        DatasetBuilder()
        .add_input('all', samples)
        .add_output('all', 'all')
        .chain()
    )

    return ds
