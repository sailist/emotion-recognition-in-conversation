"""
extract is10 1584 dimention features
"""
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
from lumo.utils import safe_io as IO
import os

res = IO.load_pkl('path/to/MOSEI.adpated.pkl')

wav_root = '/dataset/nlp/cmu-multimodal/CMU_MOSEI/Raw/Audio/Full/WAV_16000/'
wav_seg = '/dataset/nlp/cmu-multimodal/CMU_MOSEI/Raw/Audio/Full/Segment/'

os.makedirs(wav_seg, exist_ok=True)
for k, v in tqdm(res[3].items()):
    rate, sig = wavfile.read(os.path.join(wav_root, f'{k}.wav'))
    for i, (left, right) in enumerate(v.tolist()):
        left, right = abs(int(left * rate)), int(right * rate)
        sub_sig = sig[left:right]
        wavfile.write(os.path.join(wav_seg, f'{k}_{i:02d}.wav'), rate, sub_sig)

wav_seg_feature = '/dataset/nlp/cmu-multimodal/CMU_MOSEI/Raw/Audio/Full/Segment_is10/'

bin_fn = "/home/admin/opensmile-3.0.1-linux-x64/bin/SMILExtract"
config_fn = "/home/admin/opensmile-3.0.1-linux-x64/config/is09-13/IS10_paraling.conf"
for f in tqdm(os.listdir(wav_seg)):
    absf = os.path.join(wav_seg, f)
    tgtf = os.path.join(wav_seg_feature, f)
    os.system(f'{bin_fn} -C {config_fn} -I {absf} -csvoutput {tgtf}.csv')

# collect
feat_map = {}
for f in res:
    absf = os.path.join(wav_seg_feature, f)
    key = f[:-11]
    feat_map.setdefault(key, []).append(absf)

feat1_map = {}
for k, v in tqdm(list(feat_map.items())):
    features = [np.array(list(map(float, IO.load_text(vv).split('\n')[-2].replace("'unknown'", '0').split(';')))) for vv
                in v]
    feat1_map[k] = features

feat1_map = {k: np.array(v).astype(np.float32) for k, v in feat1_map.items()}

IO.dump_pkl(feat1_map, 'MOSEI.is10.pkl')
