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
from lumo.utils import safe_io as IO
from tqdm import tqdm


class FParams(Params):

    def __init__(self):
        super().__init__()
        self.mfcc = False
        self.stft = False
        self.fb = True

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


def extract_from_sig(f, intervals, kwargs: dict):
    freq, sig = wavfile.read(f)
    res = []

    for split in intervals:
        left, right = split.tolist()
        left, right = abs(int(left * freq)), int(right * freq)
        res.append(sig[left:right])

    feats = []
    for arr in res:
        if arr.ndim == 1:
            arr = arr[None, :]
        if kwargs.get('fb', False):
            feat = wav_to_fb(arr, fb_dim=40)
        elif kwargs.get('stft', False):
            feat = wav_to_stft(arr)
        elif kwargs.get('mfcc', False):
            feat = wav_to_mfcc(arr, mfcc_dim=24)
        else:
            raise NotImplementedError()
        # print(feat.shape)
        new_feat = []
        for i in range(0, feat.shape[1], 16):
            new_feat.append(feat[0, i:i + 16].mean(axis=0))
        feat = np.stack(new_feat)
        feat = feat[np.linspace(0, feat.shape[0] - 1, num=16, dtype=int)].reshape(-1)
        feats.append(feat)
    return [os.path.splitext(os.path.basename(f))[0], np.array(feats).astype(np.float32)]


def main():
    pm = FParams()

    pm.from_args()
    root = get_root('mosei')

    video_intervals = pick_datas(root, 'mosei-interval-any-7')

    wav_root = '/dataset/nlp/cmu-multimodal/CMU_MOSEI/Raw/Audio/Full/WAV_16000'

    res = []
    # newres = []
    for f in tqdm(os.listdir(wav_root)):
        interval = video_intervals.get(os.path.splitext(f)[0])
        absf = os.path.join(wav_root, f)
        if interval is not None:
            res.append([absf, interval])
            # newres.append(extract_from_sig(absf, interval, pm.to_dict()))

    executor = Parallel(n_jobs=10, verbose=10)
    newres = executor(delayed(extract_from_sig)(f, interval, pm.to_dict()) for f, interval in res)
    newres = {k: v for k, v in newres}
    IO.dump_pkl(newres, os.path.join(root, 'fbank480.pkl'))


if __name__ == '__main__':
    main()
