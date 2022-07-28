"""
"""
from lumo import DatasetBuilder
from augmentations.audio_strategies import (
    read, read_fb, read_stft, random_crop, center_crop, read_mfcc, Compose, gauss_noise
)
from .const import get_root
from .datas import pick_datas


def get_train_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    xs, xts, ys = pick_datas(root, dataset_name, split=split)

    methods = set(method.split('.'))

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('xts', xts)
        .add_input('ys', ys)
        .add_output('xts', 'xts')  # sentence
        .add_output('ys', 'ys')
    )

    feature_transform = Compose()

    if 'crop' in methods:
        feature_transform.append(random_crop(1500))

    if 'noise' in methods:
        feature_transform.append(gauss_noise())

    if 'fb' in methods:
        ds.add_output('xs', 'fb', Compose(read_fb, feature_transform))
    if 'stft' in methods:
        ds.add_output('xs', 'stft', Compose(read_stft, feature_transform))
    if 'mfcc' in methods:
        ds.add_output('xs', 'mfcc', Compose(read_mfcc, feature_transform))
    if 'raw' in methods:
        ds.add_output('xs', 'raw', read)

    return ds


def get_test_dataset(dataset_name, method='fb'):
    root = get_root(dataset_name)
    xs, xts, ys = pick_datas(root, dataset_name, split='test')
    methods = set(method.split('.'))

    ds = (
        DatasetBuilder()
        .add_idx('id')
        .add_input('xs', xs)
        .add_input('xts', xts)
        .add_input('ys', ys)
        .add_output('ys', 'ys')
        .add_output('xts', 'xts')
    )

    feature_transform = Compose()

    if 'crop' in methods:
        feature_transform.append(center_crop(1000))

    if 'fb' in methods:
        ds.add_output('xs', 'fb', Compose(read_fb, feature_transform))
    if 'stft' in methods:
        ds.add_output('xs', 'stft', Compose(read_stft, feature_transform))
    if 'mfcc' in methods:
        ds.add_output('xs', 'mfcc', Compose(read_mfcc, feature_transform))
    if 'raw' in methods:
        ds.add_output('xs', 'raw', read)

    return ds
