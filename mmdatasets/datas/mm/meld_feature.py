"""
MELD
    Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

Dataset can be downloaded from https://affective-meld.github.io/
"""
import os
from lumo.utils import safe_io as IO
import numpy as np


def meld_mmgcn_7(root, split='train', text=None):
    fn = os.path.join(root, 'MMGCN/MELD_features_raw.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids, _none
    ) = pkl

    text_fn = None
    if text == 'sbert':
        text_fn = os.path.join(root, 'MMGCN', 'sbert_map.pkl')

    if text_fn:
        video_text = IO.load_pkl(text_fn)

    res = []
    ids = train_ids if split == 'train' else test_ids
    for k in ids:
        res.append({
            'ids': video_ids[k],
            'speakers': video_speakers[k],
            'visual': video_visual[k].astype(np.float32),
            'audio': video_audio[k].astype(np.float32),
            'text': video_text[k].astype(np.float32),
            'label': video_labels[k],
            'sentence': video_sentence[k],
        })
    return res


def meld_mmgcn_text(root, *args, **kwargs):
    fn = os.path.join(root, 'MMGCN/MELD_features_raw.pkl')

    pkl = IO.load_pkl(fn)
    (
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,
        train_ids, test_ids, _none
    ) = pkl

    return video_sentence
