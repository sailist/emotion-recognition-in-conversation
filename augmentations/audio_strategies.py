import random

from scipy.io import wavfile
import numpy as np


def read(f):
    sampling_freq, audio = wavfile.read(f)  # type: int, np.ndarray
    return audio.astype(float)


def read_fb(f: str):
    return np.load(f'{f}.fb.npy').astype(float)


def read_stft(f: str):
    return np.load(f'{f}.stft.npy').astype(float)


def read_mfcc(f: str):
    return np.load(f'{f}.mfcc.npy').astype(float)


def random_crop(max_size):
    def inner(x: np.ndarray):
        size = len(x)
        if size <= max_size:
            return x
        left = np.random.randint(0, size - max_size)
        return x[left: left + max_size]

    return inner


def gauss_noise(ratio=20, p=0.5):
    def inner(x: np.ndarray):
        if random.random() < p:
            noisy = np.random.normal(0, x.max() * ratio / 100, x.shape)
            x = x + noisy
        return x

    return inner


def center_crop(max_size):
    def inner(x: np.ndarray):
        size = len(x)
        if size <= max_size:
            return x
        left = (size - max_size) // 2
        return x[left:left + max_size]

    return inner


class Compose:
    def __init__(self, *lis):
        self.lis = list(lis)

    def append(self, fwd):
        self.lis.append(fwd)

    def __call__(self, ipt):
        for fwd in self.lis:
            if fwd:
                ipt = fwd(ipt)
        return ipt
