"""
# Reference
https://github.com/libffcv/ffcv-imagenet
https://arxiv.org/abs/1906.06423
"""
from lumo import DatasetBuilder

from augmentations.image_strategies import standard, simclr, read, randaugment, basic, none
from .const import mean_std_dic, imgsize_dic, lazy_load_ds, roots
from .datas import pick_datas


def get_train_dataset(dataset_name, method='default', split='train'):
    root = roots[dataset_name]
    xs, ys = pick_datas(root, dataset_name, split=split)

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = reimg_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('ys', ys)
        .add_output('xs', 'xs', standard(mean, std, size=img_size, resize=reimg_size))
        .add_output('xs', 'sxs0', simclr(mean, std, size=img_size, resize=reimg_size))
        .add_output('xs', 'sxs1', randaugment(mean, std, size=img_size, resize=reimg_size))
        .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    return ds


def get_test_dataset(dataset_name):
    root = roots[dataset_name]
    xs, ys = pick_datas(root, dataset_name, split='test')

    mean, std = mean_std_dic.get(dataset_name, mean_std_dic.get('default'))
    img_size = imgsize_dic(dataset_name)
    assert img_size is not None

    lazy_load = dataset_name in lazy_load_ds

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('ys', ys)
        .add_output('ys', 'ys')
    )

    if lazy_load:
        ds.add_input_transform('xs', read)

    ds.add_output('xs', 'xs0', basic(mean, std, size=img_size, resize=img_size))
    ds.add_output('xs', 'xs1', none(mean, std, size=img_size))

    return ds
