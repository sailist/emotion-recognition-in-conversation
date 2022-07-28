"""
raw iemocap dataset.
contains 5 fold with Session1~5, each fold includes 2 actors.
"""
from joblib import Memory
from lumo.proc.path import cache_dir
import os
import re
from collections import Counter
from typing import Tuple, List

mem = Memory(cachedir=cache_dir())


def _get_classes(n_classes='4'):
    if n_classes == '4':
        class_names = {
            'Happiness': 0,
            'Sadness': 1,
            'Neutral': 2,
            'Anger': 3,
        }
    elif n_classes == '4.1':
        class_names = {
            'Happiness': 0,
            'Excited': 0,
            'Sadness': 1,
            'Neutral': 2,
            'Anger': 3,
        }
    elif n_classes == '6':
        class_names = {

            'Happiness': 0,
            'Sadness': 1,
            'Neutral': 2,
            'Anger': 3,
            'Excited': 4,
            'Frustration': 5,
        }
    else:
        raise NotImplementedError(n_classes)
    return class_names


def _iemocap_label_map(root, folder):
    res = []
    for f in folder:
        session_root = os.path.join(root, f, 'dialog/EmoEvaluation/Categorical/')
        fs = os.listdir(session_root)
        fs = [i for i in fs if i.endswith('txt')]

        for ff in fs:
            absf = os.path.join(session_root, ff)
            with open(absf, 'r') as r:
                lines = r.readlines()
                res.extend(lines)

    labels = {}
    names = []
    match_need = re.compile(r'(Ses.*\d) :(.*)\(')

    for name, label in [match_need.findall(r)[0] for r in res]:
        label = label.split()
        names.append(name)
        label = [i.strip(';').strip(':') for i in label]
        labels.setdefault(name, Counter()).update(label)

    labels = {k: v.most_common(1)[0][0] for k, v in labels.items()}
    return labels


def _iemocap_text_map(root, folder):
    res = []
    for f in folder:
        session_root = os.path.join(root, f, 'dialog/transcriptions/')
        fs = os.listdir(session_root)  # type: List[str]
        fs = [i for i in fs if i.endswith("txt")]

        for ff in fs:
            absf = os.path.join(session_root, ff)
            with open(absf, 'r') as r:
                lines = r.readlines()
                res.extend(lines)
    match_need = re.compile(r'(Ses.*) \[.*:(.*)')
    sents = {}

    for i, mat in enumerate([match_need.findall(r) for r in res]):
        if (len(mat)) > 0:
            name, sent = mat[0]
            sents[name] = sent.strip()
    return sents


def _iemocap_audio_map(root, folder):
    audios = {}
    for f in folder:
        transf = os.path.join(root, f, 'sentences/wav')
        for tr, _, wavs in os.walk(transf):
            for wav in wavs:
                if not (wav.endswith('wav')):
                    continue
                absf = os.path.join(tr, wav)
                audios[os.path.splitext(wav)[0]] = absf

    return audios


@mem.cache()
def iemocap_text(root, split='train') -> Tuple[List[str], List[str]]:
    """
    > site from "SMIN: Semi-supervised Multi-modal Interaction Network for Conversational Emotion Recognition"
        Since no predefined train/val/test split is provided in the IEMOCAP
        dataset, we follow the dataset split manner in previous
        works [4], [7], [36]. Specifically, dialogues from the first four
        sessions are utilized as the training set and the validation
        set. And dialogues from the last session are utilized as the
        testing set.

    Counter({'Neutral': 1726,
         'Frustration': 2916,
         'Anger': 1269,
         'Sadness': 1251,
         'Happiness': 656,
         'Excited': 1976,
         'Surprise': 110,
         'Fear': 107,
         'Other': 26,
         'Disgust': 2})

    :param root:
    :param split:
    :return:
    """
    if split == 'train':
        folder = ['Session1', 'Session2', 'Session3', 'Session4', ]
    else:
        folder = ['Session5']

    labels = _iemocap_label_map(root, folder)
    sents = _iemocap_text_map(root, folder)

    xs = []
    ys = []
    for k, v in sents.items():
        if k in labels:
            xs.append(v)
            ys.append(labels[k])

    return xs, ys


@mem.cache()
def iemocap_audio(root, split='train'):
    if split == 'train':
        folder = ['Session1', 'Session2', 'Session3', 'Session4', ]
    else:
        folder = ['Session5']

    labels = _iemocap_label_map(root, folder)
    audios = _iemocap_audio_map(root, folder)
    xs = []
    ys = []
    for k, v in audios.items():
        if k in labels:
            xs.append(v)
            ys.append(labels[k])
    return xs, ys


@mem.cache()
def iemocap_text_audio(root, split='train'):
    if split == 'train':
        folder = ['Session1', 'Session2', 'Session3', 'Session4', ]
    else:
        folder = ['Session5']

    labels = _iemocap_label_map(root, folder)
    audios = _iemocap_audio_map(root, folder)
    sents = _iemocap_text_map(root, folder)
    xs = []
    xts = []
    ys = []
    for k, v in audios.items():
        if k in labels and k in sents:
            xs.append(v)
            xts.append(sents[k])
            ys.append(labels[k])

    return xs, xts, ys


def iemocap_text_subset(n_classes='4'):
    class_names = _get_classes(n_classes)

    def inner(root, split='train'):
        xs, ys = iemocap_text(root, split)
        nxs, nys = [], []
        for x, y in zip(xs, ys):
            if y in class_names:
                nxs.append(x)
                nys.append(class_names[y])
        return nxs, nys

    return inner


def iemocap_audio_subset(n_classes='4'):
    class_names = _get_classes(n_classes)

    def inner(root, split='train'):
        xs, ys = iemocap_audio(root, split)
        nxs, nys = [], []
        for x, y in zip(xs, ys):
            if y in class_names:
                nxs.append(x)
                nys.append(class_names[y])
        return nxs, nys

    return inner


def iemocap_text_audio_subset(n_classes='4'):
    class_names = _get_classes(n_classes)

    def inner(root, split='train'):
        xs, xts, ys = iemocap_text_audio(root, split)
        nxs, nxts, nys = [], [], []
        for x, xt, y in zip(xs, xts, ys):
            if y in class_names:
                nxs.append(x)
                nxts.append(xt)
                nys.append(class_names[y])
        return nxs, nxts, nys

    return inner


def iemocap_video(root, split='train'):
    pass
