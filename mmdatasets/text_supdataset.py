"""
"""
from lumo import DatasetBuilder

from .const import mean_std_dic, get_root
from .datas import pick_datas


def get_train_dataset(dataset_name, method='default', split='train'):
    root = get_root(dataset_name)
    xs, ys = pick_datas(root, dataset_name, split=split)

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('ys', ys)
        .add_output('xs', 'xs')
        .add_output('ys', 'ys')
    )

    return ds


def get_test_dataset(dataset_name, method='default'):
    root = get_root(dataset_name)
    xs, ys = pick_datas(root, dataset_name, split='test')

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('ys', ys)
        .add_output('ys', 'ys')
    )

    ds.add_output('xs', 'xs')
    return ds
