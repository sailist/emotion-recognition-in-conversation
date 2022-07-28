"""
flattened iemocap/meld dataset and cmu-mosei/mosi dataset for multi-model emotion regognition
"""
from collections import Counter
from itertools import cycle
from lumo import DatasetBuilder

if __name__ == '__main__':
    from const import get_root
    from datas import pick_datas
else:
    from .const import get_root
    from .datas import pick_datas


def get_train_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split=split)

    if 'iemocap' in dataset_name or 'meld' in dataset_name:
        new_samples = []
        for sample in samples:
            max_len = Counter([len(v) for v in sample.values()]).most_common(1)[0][0]
            cur_samples = []
            sample = list(sample.items())
            keys = [k for k, _ in sample]
            vals = [v if len(v) == max_len else [v] * max_len for k, v in sample]
            for i in range(max_len):
                cur = {}
                for k, v in zip(keys, vals):
                    cur[k] = v[i]
                cur_samples.append(cur)
            new_samples.extend(cur_samples)
    else:
        new_samples = samples

    ds = (
        DatasetBuilder()
        .add_input('all', new_samples)
        .add_output('all', 'all')
        .subset(range(int(len(new_samples) * 0.9)))
        .chain()
    )

    return ds


def get_val_dataset(dataset_name, method='fb', split='train'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split=split)

    if 'iemocap' in dataset_name or 'meld' in dataset_name:
        new_samples = []
        for sample in samples:
            max_len = Counter([len(v) for v in sample.values()]).most_common(1)[0][0]
            cur_samples = []
            sample = list(sample.items())
            keys = [k for k, _ in sample]
            vals = [v if len(v) == max_len else [v] * max_len for k, v in sample]
            for i in range(max_len):
                cur = {}
                for k, v in zip(keys, vals):
                    cur[k] = v[i]
                cur_samples.append(cur)
            new_samples.extend(cur_samples)
    else:
        new_samples = samples

    ds = (
        DatasetBuilder()
        .add_input('all', new_samples)
        .add_output('all', 'all')
        .subset(range(int(len(new_samples) * 0.9), len(new_samples)))
        .chain()
    )

    return ds


def get_test_dataset(dataset_name, method='fb'):
    root = get_root(dataset_name)
    samples = pick_datas(root, dataset_name, split='test')

    if 'iemocap' in dataset_name or 'meld' in dataset_name:
        new_samples = []
        for sample in samples:
            max_len = Counter([len(v) for v in sample.values()]).most_common(1)[0][0]
            cur_samples = []
            sample = list(sample.items())
            keys = [k for k, _ in sample]
            vals = [v if len(v) == max_len else [v] * max_len for k, v in sample]
            for i in range(max_len):
                cur = {}
                for k, v in zip(keys, vals):
                    cur[k] = v[i]
                cur_samples.append(cur)
            new_samples.extend(cur_samples)
    else:
        new_samples = samples

    ds = (
        DatasetBuilder()
        .add_input('all', new_samples)
        .add_output('all', 'all')
        .chain()
    )

    return ds


if __name__ == '__main__':
    dataset = get_train_dataset('iemocap-cogmen-4')
    print(dataset)
