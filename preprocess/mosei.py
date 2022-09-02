"""
make CMU-MOSEI dataset from flattened format to dialogue format

download
 - http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_50/mosei_senti_data.pkl
 - http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd
 - http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip (Optional)

"""
import os
import numpy as np
import h5py
from lumo.utils import safe_io as IO
from itertools import chain

video_audio = {}
video_glove = {}
video_vision = {}

empty_vision = []
empty_audio = []
empty_text = []
invalid_time = []

CMU_MOSEI_Labels_fn = './CMU_MOSEI_Labels.csd'
mosei_senti_data_fn = 'mosei_senti_data.pkl'

raw_transcript_root = 'Raw/Transcript/Segmented/Combined/'

align = IO.load_pkl(mosei_senti_data_fn)

valid_idx = {tuple(v): i for i, v in enumerate(align['valid']['id'].tolist())}
train_idx = {tuple(v): i for i, v in enumerate(align['train']['id'].tolist())}
test_idx = {tuple(v): i for i, v in enumerate(align['test']['id'].tolist())}
sorted_valid = sorted(align['valid']['id'].tolist(), key=lambda x: (x[0], float(x[1]), float(x[2])))
sorted_train = sorted(align['train']['id'].tolist(), key=lambda x: (x[0], float(x[1]), float(x[2])))
sorted_test = sorted(align['test']['id'].tolist(), key=lambda x: (x[0], float(x[1]), float(x[2])))

for key, left, right in sorted_train:
    index = train_idx[(key, left, right)]
    right, left = float(right), float(left)
    if right - left < 0.5 or left < 0 or right < 0:
        #         print(key,left,right)
        invalid_time.append([key, left, right])
    #         continue
    vision = align['train']['vision'][index]
    audio = align['train']['audio'][index]
    text = align['train']['text'][index]

    if not vision.any():
        empty_vision.append([key, left, right])

    if not audio.any():
        empty_audio.append([key, left, right])

    if not text.any():
        empty_text.append([key, left, right])

    if vision.any():
        vision = vision[vision.any(axis=1)].mean(axis=0)
    else:
        vision = vision[0]

    if audio.any():
        audio = audio[audio.any(axis=1)].mean(axis=0)
    else:
        audio = audio[0]

    text = text[text.any(axis=1)].mean(axis=0)

    video_vision.setdefault(key, []).append(vision)
    video_audio.setdefault(key, []).append(audio)
    video_glove.setdefault(key, []).append(text)

for key, left, right in sorted_test:
    index = test_idx[(key, left, right)]
    right, left = float(right), float(left)
    if right - left < 0.5 or left < 0 or right < 0:
        #         print(key,left,right)
        invalid_time.append([key, left, right])
    #         continue
    vision = align['test']['vision'][index]
    audio = align['test']['audio'][index]
    text = align['test']['text'][index]

    if not vision.any():
        empty_vision.append([key, left, right])

    if not audio.any():
        empty_audio.append([key, left, right])

    if not text.any():
        empty_text.append([key, left, right])

    if vision.any():
        vision = vision[vision.any(axis=1)].mean(axis=0)
    else:
        vision = vision[0]

    if audio.any():
        audio = audio[audio.any(axis=1)].mean(axis=0)
    else:
        audio = audio[0]

    text = text[text.any(axis=1)].mean(axis=0)

    video_vision.setdefault(key, []).append(vision)
    video_audio.setdefault(key, []).append(audio)
    video_glove.setdefault(key, []).append(text)

for key, left, right in sorted_valid:
    index = valid_idx[(key, left, right)]
    right, left = float(right), float(left)
    if right - left < 0.5 or left < 0 or right < 0:
        invalid_time.append([key, left, right])

    vision = align['valid']['vision'][index]
    audio = align['valid']['audio'][index]
    text = align['valid']['text'][index]

    if not vision.any():
        empty_vision.append([key, left, right])

    if not audio.any():
        empty_audio.append([key, left, right])

    if not text.any():
        empty_text.append([key, left, right])

    if vision.any():
        vision = vision[vision.any(axis=1)].mean(axis=0)
    else:
        vision = vision[0]

    if audio.any():
        audio = audio[audio.any(axis=1)].mean(axis=0)
    else:
        audio = audio[0]

    text = text[text.any(axis=1)].mean(axis=0)

    video_vision.setdefault(key, []).append(vision)
    video_audio.setdefault(key, []).append(audio)
    video_glove.setdefault(key, []).append(text)

train_id = list(set([i[0] for i in sorted_train]))
valid_id = list(set([i[0] for i in sorted_valid]))
test_id = list(set([i[0] for i in sorted_test]))


def make_sentence():
    texts = {}
    for i in os.listdir(raw_transcript_root):
        if i.endswith('txt'):
            texts[i] = IO.load_text(os.path.join(raw_transcript_root, i))

    texts = {k: v.split('\n') for k, v in texts.items()}

    def splits(sent):
        key, index, left, right, uttr = sent.split('___', maxsplit=4)
        return key, index, left, right, uttr

    texts_lis = [splits(v) for v in chain(*texts.values()) if v.strip() != '']
    texts_map = {(key, float(left), float(right)): uttr for key, index, left, right, uttr in texts_lis}
    res = []
    for key, left, right in align['train']['id'].tolist():
        res.append(texts_map[(key, float(left), float(right))])
    for key, left, right in align['test']['id'].tolist():
        res.append(texts_map[(key, float(left), float(right))])
    for key, left, right in align['valid']['id'].tolist():
        res.append(texts_map[(key, float(left), float(right))])

    video_sentence = {}
    for key, left, right in chain(sorted_valid, sorted_train, sorted_test):
        video_sentence.setdefault(key, []).append(texts_map[(key, float(left), float(right))])
    return video_sentence


video_interval = {}

for key, left, right in chain(sorted_valid, sorted_train, sorted_test):
    video_interval.setdefault(key, []).append((float(left), float(right)))

raw_labels = h5py.File(CMU_MOSEI_Labels_fn)
raw_label_map = {}
for key in raw_labels['All Labels']['data'].keys():
    labels = raw_labels['All Labels']['data'][key]['features'][()]
    intervals = raw_labels['All Labels']['data'][key]['intervals'][()].tolist()
    for i in range(len(intervals)):
        left, right = intervals[i]
        raw_label_map[(key, float(left), float(right))] = labels[i]
video_emo_label = {}
for key, left, right in chain(sorted_valid, sorted_train, sorted_test):
    emo_label = raw_label_map[(key, float(left), float(right))]
    video_emo_label.setdefault(key, []).append(emo_label)

for k in chain(train_id, test_id, valid_id):
    for res in [video_interval, video_emo_label, video_audio, video_glove, video_vision]:
        res[k] = np.array(res[k]).astype(np.float32)

if raw_transcript_root is not None:
    video_sentence = make_sentence()
else:
    video_sentence = video_audio

dataset = [train_id, test_id, valid_id, video_interval, video_emo_label, video_audio, video_glove, video_vision,
           video_sentence, empty_vision, empty_audio, empty_text, invalid_time]

IO.dump_pkl(dataset, './MOSEI.adpated.pkl')
