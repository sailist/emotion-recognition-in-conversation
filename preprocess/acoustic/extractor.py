import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .melfeature import wav_to_stft, wav_to_fb, wav_to_mfcc
from scipy.io import wavfile
import os
from joblib import Parallel, delayed
from mmdatasets.datas import pick_datas
from mmdatasets.const import get_root
from lumo import Params


class FParams(Params):

    def __init__(self):
        super().__init__()
        self.mfcc = False
        self.stft = False
        self.fb = False

        self.root = True
        self.merge = True


def extract(f, kwargs: dict):
    freq, arr = wavfile.read(f)  # type: float, np.ndarray
    if arr.ndim == 1:
        arr = arr[None, :]

    res = [arr.shape[-1]]
    if kwargs.get('fb', False):
        fb_feat = wav_to_fb(arr)
        np.save(f'{f}.fb.npy', fb_feat)
        res.append(fb_feat.shape[1])
    if kwargs.get('stft', False):
        stft_feat = wav_to_stft(arr)
        np.save(f'{f}.stft.npy', stft_feat)
        res.append(stft_feat.shape[1])
    if kwargs.get('mfcc', False):
        mfcc_feat = wav_to_mfcc(arr, mfcc_dim=24)
        np.save(f'{f}.mfcc.npy', mfcc_feat)
        res.append(mfcc_feat.shape[1])

    return res


def extract_from_sig(arr, kwargs: dict):
    if arr.ndim == 1:
        arr = arr[None, :]

    if kwargs.get('fb', False):
        feat = wav_to_fb(arr)
    elif kwargs.get('stft', False):
        feat = wav_to_stft(arr)
    elif kwargs.get('mfcc', False):
        feat = wav_to_mfcc(arr, mfcc_dim=24)
    else:
        raise NotImplementedError()

    return feat


def main():
    pm = FParams()

    pm.from_args()
    root = get_root('iemocap')

    xs, ys = pick_datas(root, 'iemocap-audio-raw', 'train')
    txs, tys = pick_datas(root, 'iemocap-audio-raw', 'test')

    executor = Parallel(n_jobs=10, verbose=10)
    all_xs = [*xs, *txs]
    all_ys = [*ys, *tys]

    res = executor(delayed(extract)(f, y, pm.to_dict()) for f, y in zip(all_xs, all_ys))
    res = np.array(res)
    plt.hist(res[:, 0], bins=500)
    plt.show()
    print(res.max(axis=0))
    print(res.min(axis=0))


if __name__ == '__main__':
    main()
