"""

"""
import os
import numpy as np
from lumo.utils import safe_io as IO


def cmumosei_7(a):
    if a < -2:  # [-3, -2)
        res = 0
    elif -2 <= a and a < -1:  # [-2, -1)
        res = 1
    elif -1 <= a and a < 0:  # [-1, 0)
        res = 2
    elif 0 <= a and a <= 0:  # 0
        res = 3
    elif 0 < a and a <= 1:  # (0, 1]
        res = 4
    elif 1 < a and a <= 2:  # (1, 2]
        res = 5
    elif a > 2:  # (2, 3]
        res = 6
    else:
        raise NotImplementedError(a)
    return res


def cmumosei_2(a):
    if a < 0:
        return 0
    else:  # >= 0
        return 1


def create_emotion_label(emo_label):
    """
    binary multi-hot label matrix
    :param emo_label:
    :return:
    """
    trueLabel = []
    maxLen = []
    for j in range(emo_label.shape[0]):
        temp = np.zeros((1, 7), dtype=int)[0]
        pos = np.nonzero(emo_label[j])[0]
        if len(pos) == 0:
            temp[-1] = 1
        else:
            temp[pos] = 1
        maxLen.append(temp)
    return np.array(maxLen)
    # for i in range(emo_label.shape[0]):
    #     trueLabel.append(maxLen)
    # trueLabel = np.array(trueLabel)
    # return trueLabel


def mosei_cim(root, split='train', audio=None, label_type='emo'):
    """
    [NAACL-19-CIM](https://github.com/DushyantChauhan/NAACL-19-CIM)

    :param root:
    :param split:
    :param text:
    :return:
    """

    text = np.load(os.path.join(root, 'CIM/text.npz'))
    video = np.load(os.path.join(root, 'CIM/video.npz'))
    audio = np.load(os.path.join(root, 'CIM/audio.npz'))

    if split == 'train':
        lengths = text['train_length']
        emo_labels = text['trainEmoLabel']
        sent_labels = text['trainSentiLabel']
        text_features = text['train_data']
        video_features = video['train_data']
        audio_features = audio['train_data']
        ids = text['train_idName']
    elif split == 'val':
        lengths = text['valid_length']
        emo_labels = text['validEmoLabel']
        sent_labels = text['validSentiLabel']
        text_features = text['valid_data']
        video_features = video['valid_data']
        audio_features = audio['valid_data']
        ids = text['train_idName']
    elif split == 'test':
        lengths = text['test_length']
        emo_labels = text['testEmoLabel']
        sent_labels = text['testSentiLabel']
        text_features = text['test_data']
        video_features = video['test_data']
        audio_features = audio['test_data']
        ids = text['test_idName']
    else:
        raise NotImplementedError(f"split {split} in MOSEI")

    res = []
    # emo_labels = create_emotion_label(emo_labels)
    for i in range(len(ids)):
        length = lengths[i]

        senti2_labels = np.array([cmumosei_2(i) for i in sent_labels[i][:length, 0]])
        senti7_labels = np.array([cmumosei_7(i) for i in sent_labels[i][:length, 0]])
        emo_label = create_emotion_label(emo_labels[i][:length])

        audio_feature = audio_features[i][:length].astype(np.float32)
        if audio == 'pad80':
            audio_feature = np.concatenate([audio_feature, np.zeros(audio_feature.shape[0], 6)], axis=-1)

        res.append({
            'ids': ids[i],
            'length': lengths[i],
            'speakers': [0],
            'visual': video_features[i][:length].astype(np.float32),
            'audio': audio_feature,
            'text': text_features[i][:length].astype(np.float32),
            'label': senti2_labels,
            'emo_label': emo_label,
            'senti2_label': senti2_labels,
            'senti7_label': senti7_labels,
        })

    return res


def mosei_adapted(root, split, audio=None, text=None, label_type='emo', balance=False):
    fn = os.path.join(root, 'MOSEI.adpated.pkl')
    pkl = IO.load_pkl(fn)

    (train_id, test_id, valid_id,
     video_interval,
     video_emo_label, video_audio, video_text, video_vision,
     video_sentence,
     empty_vision, empty_audio, empty_text, invalid_time) = pkl

    ids = train_id if split == 'train' else test_id if split == 'test' else valid_id  # type: list

    if balance and split == 'train':
        balance_train_id = IO.load_pkl(os.path.join(root, 'balanced_train_id.pkl'))
        ids.extend(balance_train_id)

    text_fn = None
    if text == 'sbert':
        text_fn = os.path.join(root, 'sbert_map.pkl')
    if text_fn:
        video_text = IO.load_pkl(text_fn)

    audio_fn = None
    if audio == 'fbank':
        audio_fn = os.path.join(root, 'fbank480.pkl')
    elif audio == 'is10':
        audio_fn = os.path.join(root, 'MOSEI.is10.pkl')

    if audio_fn:
        video_audio = IO.load_pkl(audio_fn)

    res = []
    count = 0
    droped = 0

    for k in ids:
        # [sentiment, happy,sad,anger,surprise,disgust,fear]

        if label_type == 'emo':

            label = video_emo_label[k][:, 1:].argmax(axis=-1)
        elif label_type == 'multi':
            label = video_emo_label[k]
        elif label_type == 'sent_2':
            label = np.array([cmumosei_2(i) for i in video_emo_label[k][:, 0].tolist()], dtype=int)
        elif label_type == 'sent_2+':
            if not video_emo_label[k][:, 0].any():
                droped += len(video_emo_label[k])
                continue
            label = np.array([cmumosei_2(i) for i in video_emo_label[k][:, 0].tolist()], dtype=int)
        elif label_type == 'sent_7':
            label = np.array([cmumosei_7(i) for i in video_emo_label[k][:, 0].tolist()], dtype=int)
        else:
            raise NotImplementedError(label_type)

        senti2_labels = np.array([cmumosei_2(i) for i in video_emo_label[k][:, 0]])
        senti7_labels = np.array([cmumosei_7(i) for i in video_emo_label[k][:, 0]])
        emo_labels = create_emotion_label(video_emo_label[k][:, 1:])

        visual_data = video_vision[k]
        audio_data = video_audio[k]
        text_data = video_text[k]
        sentence_data = video_sentence[k]

        if label_type == 'sent_2+':
            mask = video_emo_label[k][:, 0] != 0
            droped += (mask == False).sum()
            if mask.any():
                visual_data = visual_data[mask]
                audio_data = audio_data[mask]
                text_data = text_data[mask]
                sentence_data = np.array(sentence_data)[mask].tolist()
                label = label[mask]
                emo_labels = emo_labels[mask]
                senti2_labels = senti2_labels[mask]
                senti7_labels = senti7_labels[mask]

        count += len(visual_data)

        data = {
            'ids': k,
            'label': label,
            'speakers': [[0]],
            'visual': visual_data,
            'audio': audio_data,
            'text': text_data,
            'sentence': sentence_data,
            'emo_label': emo_labels,
            'senti2_label': senti2_labels,
            'senti7_label': senti7_labels,
        }

        res.append(data)
    print(f'create {count} uttrs, droped {droped}')
    return res


def mosei_interval(root, *args, **kwargs):
    fn = os.path.join(root, 'MOSEI.adpated.pkl')
    pkl = IO.load_pkl(fn)

    (train_id, test_id, valid_id,
     video_interval,
     video_emo_label, video_audio, video_glove, video_vision,
     video_sentence,
     empty_vision, empty_audio, empty_text, invalid_time) = pkl

    return video_interval


def mosei_text(root, *args, **kwargs):
    fn = os.path.join(root, 'MOSEI.adpated.pkl')
    pkl = IO.load_pkl(fn)

    (train_id, test_id, valid_id,
     video_interval,
     video_emo_label, video_audio, video_glove, video_vision,
     video_sentence,
     empty_vision, empty_audio, empty_text, invalid_time) = pkl

    return video_sentence
