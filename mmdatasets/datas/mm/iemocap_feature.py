"""
raw iemocap dataset.
feature
"""
import numpy as np
from joblib import Memory
from lumo.proc.path import cache_dir
import os
import re
from collections import Counter
from typing import Tuple, List, Dict
import h5py
from lumo.utils import safe_io as IO


def label_cogmen(root):
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_labels[k]):
            res[xx] = yy
    return res


def trainsplit_cogmen(root):
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids,
    ) = pkl

    return train_ids


def testsplit_cogmen(root):
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids,
    ) = pkl

    return test_ids


def visual_openface(root):
    """
    used in
    - COGMEN COntextualized GNN based Multimodal Emotion recognitioN
    > video features (size 512) are taken from Baltrusaitis et al. (2018), Openface 2.0: Facial behavior analysis toolkit.,

    shape [7433, 512]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_visual[k]):
            res[xx] = yy
    return res


def audio_opensmile_cogmen(root) -> Dict[str, np.ndarray]:
    """
    used in
    - COGMEN COntextualized GNN based Multimodal Emotion recognitioN
    > audio features (size 100) are extracted using OpenSmile (Eyben et al., 2010),

    shape [7433, 100]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_audio[k]):
            res[xx] = yy
    return res


def text_sroberta(root):
    """
    used in
    - COGMEN COntextualized GNN based Multimodal Emotion recognitioN
    > text features (size 768) are extracted using sBERT (Reimers and Gurevych, 2019).

    shape: [7433, 100]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')
    sbert_fn = os.path.join(root, 'sbert_map.pkl')

    pkl = IO.load_pkl(fn)
    video_sbert = IO.load_pkl(sbert_fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_sbert[k]):
            res[xx] = yy
    return res


def text_cogmen(root, n_class=4, *args, **kwargs):
    """
    used in
    - COGMEN COntextualized GNN based Multimodal Emotion recognitioN
    > text features (size 768) are extracted using sBERT (Reimers and Gurevych, 2019).

    shape: [7433, 100]
    :param root:
    :return:
    """
    if n_class == 4:
        fn = os.path.join(root, 'cogmen/iemocap_4/IEMOCAP_features_4.pkl')
    else:
        fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    return video_sentence


def video_cogmen(root, n_class=4, *args, **kwargs):
    """
    used in
    - COGMEN COntextualized GNN based Multimodal Emotion recognitioN
    > text features (size 768) are extracted using sBERT (Reimers and Gurevych, 2019).

    shape: [7433, 100]
    :param root:
    :return:
    """
    if n_class == 4:
        fn = os.path.join(root, 'cogmen/iemocap_4/IEMOCAP_features_4.pkl')
    else:
        fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    texts = []
    avis = {}

    for folder in ['Session1/', 'Session2/', 'Session3/', 'Session4/', 'Session5/', ]:
        dirfn = os.path.join(root, 'IEMOCAP_full_release', folder, 'dialog/transcriptions')
        for transfn in os.listdir(dirfn):
            if transfn.endswith('txt') and not transfn.startswith('.'):
                txt = IO.load_text(os.path.join(dirfn, transfn)).split('\n')
                texts.extend([transfn + ' ' + i for i in txt])
        avidirfn = os.path.join(root, 'IEMOCAP_full_release', folder, 'dialog/avi/DivX')
        for avifn in os.listdir(avidirfn):
            if avifn.endswith('avi') and not avifn.startswith('.'):
                avis[avifn.split('.')[0]] = os.path.join(avidirfn, avifn)

    match = re.compile('(.*txt) (Ses.*) \[([0-9.]+)\-([0-9.]+)\]: (.*)')
    matched = [match.findall(i) for i in texts]
    matched = [i[0] for i in matched if len(i) > 0]
    match_dict = {}
    for ufn, uid, left, right, sent in matched:
        ufn = ufn.split('.')[0]
        match_dict.setdefault(ufn, []).append([uid, float(left), float(right), sent])

    aligned_match_dict = {}
    for k, old_sents in video_sentence.items():
        if k not in match_dict:
            continue
        raw_sent = match_dict[k]

        b_iter = iter(raw_sent)
        new_b = []
        for a in old_sents:
            ufn, left, right, b = next(b_iter)
            while b != a:
                ufn, left, right, b = next(b_iter)
            new_b.append([ufn, left, right])
        assert len(new_b) == len(old_sents)
        aligned_match_dict[k] = new_b

    res = {}
    for k in match_dict:
        res[k] = {
            'timestamp': aligned_match_dict[k],
            'speaker': video_speakers[k],
            'fn': avis[k],
            'video_sentence': video_sentence[k],
            'video_labels': video_labels[k],
        }
    return res


def visual_densenet_ferp(root):
    """
    used in
    - MMGCN Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation
    > The visual facial expression features are extracted using a DenseNet (Huang et al., 2015)
    pre-traind on the Facial Expression Recognition Plus (FER+) corpus (Barsoum et al., 2016).

    shape [7433, 342]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'MMGCN/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_visual[k]):
            res[xx] = yy
    return res


def audio_opensmile_is10(root) -> Dict[str, np.ndarray]:
    """
    used in
    - MMGCN Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation
    > The acoustic raw features are extracted using the OpenSmile toolkit with IS10 configuration (Schuller et al., 2011).

    shape [7433, 1582]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'MMGCN/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_audio[k]):
            res[xx] = yy
    return res


def text_textcnn(root):
    """
    used in
    - MMGCN Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation
    > The textual raw features are extracted using TextCNN following (Hazarika et al., 2018a).

    shape [7433, 100]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'MMGCN/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        _, _,
    ) = pkl

    res = {}
    for k, v in video_ids.items():
        for xx, yy in zip(v, video_text[k]):
            res[xx] = yy
    return res


def visual_densenet_fer(root):
    """
    used in
    - Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities
    > We extract the facial expres- sion features using a pretrained DenseNet (Huang et al., 2017)
    which is trained based on the Facial Expression Recognition Plus (FER+) corpus (Barsoum et al., 2016).
    We denote the facial expression features as “Denseface”. The “Denseface” are frame-level sequential
    features based on the detected faces from the video frames, and the feature vectors are in 342 dimensions.

    shape: [5531, 50, 342]
    :param root:
    :return:
    """

    fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/V/denseface.h5')
    h5f = h5py.File(fn)
    return {k: h5f[k][()]
            for k in h5f.keys()}


def audio_opensmile_is13_t(root):
    """
    used in
    - Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities
    > OpenSMILE toolkit (Eyben et al., 2010) with the configuration of “IS13 ComParE”
    is used to extract frame-level features, which have similar performance with the IS10 utterance-level acoustic features
    used in (Liang et al., 2020). We denote the features as “ComParE” and the feature vectors are in 130 dimensions.

    shape [5531, t, 130]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/A/comparE.h5')
    h5f = h5py.File(fn)
    return {k: h5f[k][()]
            for k in h5f.keys()}


def text_bert_large(root):
    """
    used in
    - Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities
    > We extract contextual word embeddings using a pretrained BERT-large model (Devlin et al., 2019)
    which is one of the state-of- the-art language representations.
    We denote the word embeddings as “Bert” and the features are in 1024 dimensions.

    shape: [5531, 22, 1024]
    :param root:
    :return:
    """
    fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/L/bert_large.h5')
    h5f = h5py.File(fn)
    return {k: h5f[k][()]
            for k in h5f.keys()}


def iemocap_cogmen_6(root, split='train', text='', visual=''):
    fn = os.path.join(root, 'cogmen/iemocap/IEMOCAP_features.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids,
    ) = pkl

    # replace textual feature
    text_fn = None
    if text == 'robert':
        text_fn = os.path.join(root, 'cogmen/iemocap/robert_map.pkl')
    elif text == 'sbert':
        text_fn = os.path.join(root, 'cogmen/iemocap/sbert_map.pkl')

    if text_fn:
        video_text = IO.load_pkl(text_fn)

    visual_fn = None
    if 'tsn' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap/tsn_vfeat.pkl')
    elif 'tsnss' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap/tsn_vfeat_ss.pkl')
    elif 'x3d' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap/x3d_vfeat.pkl')

    if visual_fn:
        _video_visual = IO.load_pkl(visual_fn)
        if '+' in visual:
            for k in _video_visual:
                video_visual[k] = np.concatenate([video_visual[k], _video_visual[k]], axis=1)
        else:
            video_visual = _video_visual

    res = []
    ids = train_ids if split == 'train' else test_ids

    for k in ids:
        res.append({
            'ids': video_ids[k],
            'speakers': [[1, 0] if i == 'M' else [0, 1] for i in video_speakers[k]],
            'visual': video_visual[k],
            'audio': video_audio[k],
            'text': video_text[k],
            'label': video_labels[k],
            'sentence': video_sentence[k],
        })
    return res


def iemocap_cogmen_4(root, split='train', text='', visual=''):
    fn = os.path.join(root, 'cogmen/iemocap_4/IEMOCAP_features_4.pkl')
    pkl = IO.load_pkl(fn)

    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids,
    ) = pkl

    # replace textual feature
    text_fn = None
    if text == 'robert':
        text_fn = os.path.join(root, 'cogmen/iemocap_4/robert_map.pkl')
    elif text == 'sbert':
        text_fn = os.path.join(root, 'cogmen/iemocap_4/sbert_map.pkl')

    if text_fn:
        video_text = IO.load_pkl(text_fn)

    visual_fn = None
    if 'tsn' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap_4/tsn_vfeat.pkl')
    elif 'tsnss' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap_4/tsn_vfeat_ss.pkl')
    elif 'x3d' in visual:
        visual_fn = os.path.join(root, 'cogmen/iemocap_4/x3d_vfeat.pkl')

    if visual_fn:
        _video_visual = IO.load_pkl(visual_fn)
        if '+' in visual:
            for k in _video_visual:
                video_visual[k] = np.concatenate([video_visual[k], _video_visual[k]], axis=1)
        else:
            video_visual = _video_visual

    res = []
    ids = train_ids if split == 'train' else test_ids
    for k in ids:
        res.append({
            # 'ids': video_ids[k],
            'speakers': [[1, 0] if i == 'M' else [0, 1] for i in video_speakers[k]],
            'visual': video_visual[k],
            'audio': video_audio[k],
            'text': video_text[k],
            'label': video_labels[k],
            'sentence': video_sentence[k],
        })
    return res


def iemocap_mmin_4(root, split='train'):
    v = visual_densenet_fer(root)
    a = audio_opensmile_is13_t(root)
    t = text_bert_large(root)
    if split == 'train':
        label_fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', 'trn_label.npy')
        int2name_path = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', f"trn_int2name.npy")
    elif split == 'val':
        label_fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', 'val_label.npy')
        int2name_path = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', f"val_int2name.npy")
    else:
        label_fn = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', 'tst_label.npy')
        int2name_path = os.path.join(root, 'MMIN2021/IEMOCAP_features_2021/target', f'1', f"tst_int2name.npy")
    label = np.load(label_fn)
    label = np.argmax(label, axis=1)
    int2name = [i[0].decode() for i in np.load(int2name_path).tolist()]

    res = []
    for i, name in enumerate(int2name):
        res.append({
            'visual_feature': v[name],
            'text_feature': t[name],
            'audio_feature': a[name],
            'label': label[i],
            'name': name,
        })
    return res


def iemocap_mmin_6(root, split='train'):
    pass
