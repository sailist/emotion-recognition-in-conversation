"""
"""
from collections import Counter

import numpy as np
from torch.utils.data import RandomSampler, WeightedRandomSampler
from datetime import datetime
from typing import ClassVar, Union
import json
import torch

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, MetricType, TrainStage, Record, CollateBase
from lumo.data import DataLoaderType
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from lumo.contrib.torch.tensor import onehot
from models.module_utils import ModelParams
from mmdatasets.dataset_utils import DataParams
from torch.nn import functional as F
from dbrecord import PList


class MMBaseParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.class_names = []
        self.modality = self.choice('atv', 'av', 'at', 'tv', 't', 'a', 'v')

        self.n_speakers = 2

        # input controll
        self.batch_first = True
        self.speaker_onehot = False

        self.balance_data = False

        self.evalute_stage = False

        self.hidden_text = 100
        self.hidden_audio = 100
        self.hidden_visual = 100
        self.hidden_all = 300
        self.reimplement = False

        self.mosei_metric = 'multiemo'

        # metric
        self.confusion_matrix = True

    def iparams(self):
        super(MMBaseParams, self).iparams()
        if self.get('debug'):
            self.train.batch_size = 2
            self.test.batch_size = 2
            self.train.num_workers = 0
            self.test.num_workers = 0

        if 'mosei' not in self.dataset:
            self.mosei_metric = ''

        if 'iemocap' in self.dataset:
            if self.n_classes == 4:
                self.class_names = ['hap', 'sad', 'neu', 'ang']
            elif self.n_classes == 6:
                self.class_names = ['hap',
                                    'sad',
                                    'neu',
                                    'ang',
                                    'exc',
                                    'fru']
            if 'cogmen' in self.dataset:
                self.hidden_audio = 100
                self.hidden_text = 100
                self.hidden_visual = 512

        elif 'meld' in self.dataset:
            self.class_names = [
                'neutral', 'sad', 'mad', 'scared', 'powerful', 'peaceful', 'joyful'
            ]
            self.n_speakers = 9
            if 'mmgcn' in self.dataset:
                self.hidden_audio = 300
                self.hidden_text = 600
                self.hidden_visual = 342
        elif 'mosei' in self.dataset:
            self.class_names = ["hap", "sad", "disgust", "fear", "surprise", "ang"]
            self.hidden_text = 300
            self.hidden_audio = 74
            self.hidden_visual = 35

        if 'pad80' in self.dataset:
            self.hidden_audio = 80
        elif 'fbank' in self.dataset:
            self.hidden_audio = 640
        elif 'is10' in self.dataset:
            self.hidden_audio = 1584

        # modified/reextracted text feature
        if 'sbert' in self.dataset or 'robert' in self.dataset:
            self.hidden_text = 768

        # modified/reextracted visual feature
        hv = None
        if 'tsn' in self.dataset:
            hv = 2048

        if hv:
            if 'v+' in self.dataset:  # concat with original visual feature
                self.hidden_visual += hv
            else:
                self.hidden_visual = hv

        self.hidden_all = 0
        if 't' in self.modality:
            self.hidden_all += self.hidden_text
        if 'a' in self.modality:
            self.hidden_all += self.hidden_audio
        if 'v' in self.modality:
            self.hidden_all += self.hidden_visual

        # if self.device == 'cpu':
        #     self.train.batch_size = 4


ParamsType = MMBaseParams


class MMBaseTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=-1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)

        self.pred_info = PList(self.exp.blob_file('predicton.sqlite'))
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def to_mosei_multitask_logits(self, xs):
        raise NotImplementedError()

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_ema_logits(self, xs):
        raise NotImplementedError()

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if logits.ndim == 3:
            logits = logits[batch['attention_mask'].bool()]

        meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def mosei_test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        emo_label = batch['emo_label']
        ys = batch['senti2_label']
        logits2, logits7 = self.to_mosei_multitask_logits(batch)
        if params.get('confusion_matrix', False):
            attention_mask = (ys >= 0)
            self.true.extend(ys[attention_mask].cpu().numpy().tolist())
            self.pred.extend(logits2[attention_mask].argmax(dim=-1).cpu().numpy().tolist())

            self.true_multi.extend(emo_label.cpu().numpy())
            self.pred_multi.extend(torch.sigmoid(logits7).detach().cpu().numpy())

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        if params.get('mosei_metric', 'multiemo') == 'multiemo':
            return self.mosei_test_step(batch, params)

        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if logits.ndim == 3:
            logits = logits[batch['attention_mask'].bool()]

        if params.get('confusion_matrix', False):
            attention_mask = (ys >= 0)
            self.true.extend(ys[attention_mask].cpu().numpy().tolist())
            self.pred.extend(logits[attention_mask].argmax(dim=-1).cpu().numpy().tolist())

        meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]

        return meter

    def on_process_loader_begin(self, trainer: Trainer, func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f'set seed {params.seed}')

    def on_train_epoch_end(self, trainer: Trainer, func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, record, *args, **kwargs)
        self.metric_board.append(dict(record.agg()), step=self.eidx, stage='train')

    def on_eval_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        super().on_eval_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confusion_matrix', False):
                self.pred = []
                self.true = []

    def on_test_begin(self, trainer: Trainer, func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confusion_matrix', False):
                self.pred = []
                self.true = []

                self.pred_multi = []
                self.true_multi = []

    def weighted_accuracy(self, y_true, y_pred):
        TP, TN, FN, FP, N, P = 0, 0, 0, 0, 0, 0
        for i, j in zip(y_true, y_pred):
            if i == 1 and i == j:
                TP += 1
            elif i == 0 and i == j:
                TN += 1

            if i == 1 and i != j:
                FN += 1
            elif i == 0 and i != j:
                FP += 1

            if i == 1:
                P += 1
            else:
                N += 1

        w_acc = (1.0 * TP * (N / (1.0 * P)) + TN) / (2.0 * N)

        return w_acc, TP, TN, FP, FN, P, N

    def on_test_end(self, trainer: Trainer, func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        if self.is_main:
            if params.get('confusion_matrix', False):
                from sklearn import metrics
                if len(self.pred) > 0:
                    cm = metrics.confusion_matrix(self.true, self.pred, labels=range(params.n_classes))
                    self.logger.raw(cm)

                    cls_pre, cls_rec, cls_f1, _ = metrics.precision_recall_fscore_support(
                        self.true, self.pred
                    )
                    # cls_pre = {k: v for k, v in zip(params.class_names, cls_pre)}
                    # cls_rec = {k: v for k, v in zip(params.class_names, cls_rec)}
                    # cls_f1 = {k: v for k, v in zip(params.class_names, cls_f1)}

                    accuracy = metrics.accuracy_score(self.true, self.pred)
                    wa = metrics.balanced_accuracy_score(self.true, self.pred)
                    precision = metrics.precision_score(self.true, self.pred, average='weighted')
                    recall = metrics.recall_score(self.true, self.pred, average='weighted')
                    wf1 = metrics.f1_score(self.true, self.pred, average='weighted')
                    mif1 = metrics.f1_score(self.true, self.pred, average='micro')
                    maf1 = metrics.f1_score(self.true, self.pred, average='macro')

                    if len(self.true_multi) > 0:
                        true_multi = np.array(self.true_multi)
                        t = 0.5
                        accs = []
                        f1s = []
                        waccs = []
                        pred_multi = np.array(self.pred_multi)
                        self.logger.raw(metrics.classification_report(true_multi, (pred_multi > t).astype(int)))
                        for i in range(7):
                            column = (pred_multi[:, i] > t).astype(int)
                            accs.append(metrics.accuracy_score(true_multi[:, i], column))
                            f1s.append(metrics.precision_recall_fscore_support(true_multi[:, i], column,
                                                                               average='weighted')[2])
                            w_acc, TP, TN, FP, FN, P, N = self.weighted_accuracy(true_multi[:, i], column)
                            waccs.append(w_acc)
                        self.logger.info('with thresh ', t)
                        self.logger.info('cls acc', " ".join([f"{float(i):.3f}" for i in accs]))
                        self.logger.info('cls f1', " ".join([f"{float(i):.3f}" for i in f1s]))
                        self.logger.info('cls wa', " ".join([f"{float(i):.3f}" for i in waccs]))
                        self.logger.info('acc', np.mean(accs), 'f1', np.mean(f1s), 'wa', np.mean(waccs))

                        # index = np.argmax(accs)
                        # pred_multi = np.array(self.pred_multi)
                        # self.logger.raw(
                        #     metrics.classification_report(self.true_multi, (pred_multi > th[index]).astype(int)))

                    m = Meter()

                    with self.database:
                        m.update(self.database.update_metric_pair('pre', precision, 'cls_pre', cls_pre, compare='max'))
                        m.update(self.database.update_metric_pair('rec', recall, 'cls_rec', cls_rec, compare='max'))
                        m.update(self.database.update_metric_pair('f1', wf1, 'cls_f1', cls_f1, compare='max'))
                        m.update(self.database.update_metrics(dict(acc=accuracy,
                                                                   wa=wa,
                                                                   mif1=mif1,
                                                                   maf1=maf1),
                                                              compare='max'))

                    self.metric_board.append({
                        **m.todict(),
                        **record.agg(),
                        "cm": cm,
                    }, step=self.eidx, stage='test')
                    self.database.flush()
                    self.logger.info('Best Results', m)
                    self.pred_info.append([self.true, self.pred])
                    self.pred_info.flush()

    def save_best_model(self):
        file = self.exp.blob_file('best_model.ckpt', 'models')
        torch.save(self.state_dict(), file)
        self.logger.info(f'saved best model at {file}')

    def save_last_model(self):
        file = self.exp.blob_file('last_model.ckpt', 'models')
        torch.save(self.state_dict(), file)
        self.logger.info(f'saved last model at {file}')

    @classmethod
    def generate_exp_name(cls) -> str:
        return f'{cls.dirname()}.{cls.filebasename()}'

    def on_train_end(self, trainer: Trainer, func, params: ParamsType, record: Record, *args, **kwargs):
        super().on_train_end(trainer, func, params, record, *args, **kwargs)
        self.database.flush()


class ERCCollate(CollateBase):
    def __init__(self, params: ParamsType):
        super().__init__(params)
        self.batch_first = params.batch_first
        self.speaker_onehot = params.speaker_onehot
        self.n_classes = params.n_classes
        self.n_speakers = params.n_speakers
        self.modalities = params.modality
        self.speaker_to_idx = {"M": 0, "F": 1}

    def __call__(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s['text']) for s, in samples]).long()
        mx = torch.max(text_len_tensor).item()

        # attention_mask
        attention_mask = torch.zeros(len(samples), mx)
        for i, item in enumerate(text_len_tensor):
            attention_mask[i, :item] = 1

        input_tensor = []
        text_feature = []
        audio_feature = []
        visual_feature = []

        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        multi_emo_labels = []
        senti2_labels = []
        utterances = []
        for i, (dic,) in enumerate(samples):
            sentence = dic.get('sentence')
            speakers = dic['speakers']

            visual = dic['visual']
            audio = dic['audio']
            textual = dic['text']
            label = dic['label']
            emo_label = dic.get('emo_label')
            senti2_label = dic.get('senti2_label')

            cur_len = len(textual)

            if sentence is not None:
                utterances.append(sentence)

            # construct each modality feature
            tmp_a = []
            tmp_v = []
            tmp_t = []

            for t, a, v in zip(textual, audio, visual):
                tmp_a.append(torch.from_numpy(a))
                tmp_v.append(torch.from_numpy(v))
                tmp_t.append(torch.from_numpy(t))

            modality_features = [audio_feature, visual_feature, text_feature]
            for ii, tmp in enumerate([tmp_a, tmp_v, tmp_t]):
                tmp = torch.stack(tmp)
                tmp = torch.cat([tmp, torch.zeros(mx - len(tmp), tmp.shape[1], dtype=tmp.dtype)])
                modality_features[ii].append(tmp)

            # construct merged feature
            tmp = []
            for t, a, v in zip(textual, audio, visual):
                res = {'t': t, 'a': a, 'v': v}
                nres = torch.cat([torch.from_numpy(res[i]) for i in self.modalities])
                tmp.append(nres)

            tmp = torch.stack(tmp)
            tmp = torch.cat([tmp, torch.zeros(mx - len(tmp), tmp.shape[1], dtype=tmp.dtype)])
            input_tensor.append(tmp)

            # create speaker tensor
            speaker_tensor[i, :cur_len] = torch.tensor(speakers).argmax(dim=-1)

            # create label
            labels.extend(label)
            if emo_label is not None:
                multi_emo_labels.extend(torch.from_numpy(emo_label))
            if senti2_label is not None:
                senti2_labels.extend(torch.from_numpy(senti2_label))

        stack_dim = 0 if self.batch_first else 1

        if not self.batch_first:
            speaker_tensor = speaker_tensor.transpose(0, 1)

        if self.speaker_onehot:
            speaker_tensor = onehot(speaker_tensor, self.n_speakers)

        label = torch.tensor(labels).long()
        data = {
            'attention_mask': attention_mask,
            "text_length": text_len_tensor,
            "text_feature": torch.stack(text_feature, dim=stack_dim) if 't' in self.modalities else None,
            "audio_feature": torch.stack(audio_feature, dim=stack_dim) if 'a' in self.modalities else None,
            "visual_feature": torch.stack(visual_feature, dim=stack_dim) if 'v' in self.modalities else None,
            "input_tensor": torch.stack(input_tensor, dim=stack_dim),
            "speaker_tensor": speaker_tensor,
            "label": label,

        }
        if len(utterances) > 0:
            data["utterance_texts"] = utterances

        if len(multi_emo_labels) > 0:
            data["emo_label"] = torch.stack(multi_emo_labels)
        if len(senti2_labels) > 0:
            data["senti2_label"] = torch.stack(senti2_labels)

        return data


class ERCDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = ERCCollate(params)
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   method=params.get('method'))

            dl = ds.DataLoader(**params.train.to_dict(), collate_fn=collate_fn)
        # elif stage.is_val():
        #     ds = get_val_dataset(params.dataset,
        #                          method=params.get('method'))
        #     dl = ds.DataLoader(**params.val.to_dict(), collate_fn=collate_fn)

        else:
            ds = get_test_dataset(params.dataset,
                                  method=params.get('method'))
            dl = ds.DataLoader(**params.test.to_dict(), collate_fn=collate_fn)
        print(stage, ds)

        self.regist_dataloader_with_stage(stage, dl)


def main(
        trainer_cls: ClassVar[Trainer],
        params_cls: ClassVar[ParamsType],
        dm: ClassVar[DataModule] = ERCDM,
):
    params = params_cls()
    params.from_args()

    dm = dm(params)
    trainer = trainer_cls(params, dm)

    if params.get('eval_first', False):
        trainer.test()

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_model()
