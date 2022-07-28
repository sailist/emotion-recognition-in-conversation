import numpy as np
from torch._six import string_classes
import collections.abc
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from lumo.data.collate import CollateBase
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Wav2Vec2Processor

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        try:
            return torch.stack(batch, 0, out=out)
        except:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class TokenizerCollate(CollateBase):
    def __init__(self, tokenizer: PreTrainedTokenizer, keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def after_collate(self, batch):
        for key in self.keys:
            batch[key] = self.tokenizer(
                batch[key],
                return_tensors='pt',
                padding=True)
        return batch


class AudioProcessCollate(CollateBase):

    def __init__(self, processor: Wav2Vec2Processor, keys, *args, **kwargs):
        super().__init__(default_collate, *args, **kwargs)
        self.processor = processor
        if isinstance(keys, str):
            keys = [keys]
        self.keys = set(keys)

    def before_collate(self, sample_list):

        return super().before_collate(sample_list)

    def collate(self, sample_list):
        return super().collate(sample_list)

    def after_collate(self, batch):
        for key in self.keys:
            batch[key] = self.processor(
                [i.numpy() for i in batch[key]],
                sampling_rate=16000,
                return_tensors='pt',
                padding=True)
        return batch


class AudioPaddingProcessCollate(CollateBase):

    def __init__(self, keys, max_length=2000, padding='strip', *args, **kwargs):
        super().__init__(default_collate, *args, **kwargs)
        if isinstance(keys, str):
            keys = [keys]
        self.max_length = max_length
        self.keys = set(keys)
        self.padding = padding

    def before_collate(self, sample_list):
        for key in self.keys:
            if key not in sample_list[0]:
                continue

            if self.padding == 'strip':
                max_length = max([i[key].squeeze().shape[0] for i in sample_list])
                if self.max_length is not None:
                    max_length = min(max_length, self.max_length)
            elif self.padding == 'max_length':
                max_length = self.max_length
            else:
                raise NotImplementedError()

            for dic in sample_list:
                feat = dic[key].squeeze()
                if feat.shape[0] > max_length:
                    dic[key] = {
                        'input_values': feat[:max_length],
                        'attention_mask': np.ones(max_length)
                    }
                else:
                    dic[key] = {
                        'input_values': np.concatenate([
                            feat,
                            np.zeros([max_length - feat.shape[0], feat.shape[1]], dtype=float)
                        ]),
                        'attention_mask': np.concatenate([np.ones(len(feat)), np.zeros(max_length - feat.shape[0])])
                    }

        return super().before_collate(sample_list)


class ComposeCollate(CollateBase):
    def __init__(self, collate_fns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fns = collate_fns

    def before_collate(self, sample_list):
        if self.collate_fns:
            for fn in self.collate_fns:
                sample_list = fn.before_collate(sample_list)
        return sample_list

    def after_collate(self, batch):
        if self.collate_fns:
            for fn in self.collate_fns:
                batch = fn.after_collate(batch)
        return batch
