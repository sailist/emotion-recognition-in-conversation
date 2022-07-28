"""
Multi-modal emotion recognition conversation dataset (iemocap, meld)
"""
from lumo import DatasetBuilder
from .const import get_root
from .datas import pick_datas


def get_train_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split=split)

    ds = (
        DatasetBuilder()
        .add_input('all', samples)
        .add_output('all', 'all')
        .subset(range(int(len(samples) * 0.9)))
        .chain()
    )

    return ds


def get_val_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split=split)

    ds = (
        DatasetBuilder()
        .add_input('all', samples)
        .add_output('all', 'all')
        .subset(range(int(len(samples) * 0.9), len(samples)))
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
