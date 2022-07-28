"""
"""
from datetime import datetime
from typing import ClassVar, Union
import json
import torch

from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, MetricType, TrainStage, Record
from lumo.data import DataLoaderType

from models.module_utils import ModelParams
from mmdatasets.dataset_utils import DataParams
from torch.nn import functional as F
from dbrecord import PList

from contrib.database import TableRow


class MMBaseParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.module = None
        self.method = None

        self.class_names = []
        self.modality = self.choice('t', 'atv', 'av', 'at', 'tv', 't', 'a', 't')

        self.n_speakers = 2

        self.hidden_text = 100
        self.hidden_audio = 100
        self.hidden_visual = 100
        self.hidden_all = 300
        self.reimplement = False

    def iparams(self):
        super(MMBaseParams, self).iparams()
        if self.get('debug'):
            self.train.batch_size = 2
            self.test.batch_size = 2
            self.train.num_workers = 0
            self.test.num_workers = 0

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

    def to_logits(self, xs):
        raise NotImplementedError()

    def to_ema_logits(self, xs):
        raise NotImplementedError()

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if logits.ndim == 3:
            meter.sum.Lall = F.cross_entropy(logits.permute(0, 2, 1), ys)
        else:
            meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if params.get('confuse_matrix', False):
            attention_mask = (ys >= 0)
            self.true.extend(ys[attention_mask].cpu().numpy().tolist())
            self.pred.extend(logits[attention_mask].argmax(dim=-1).cpu().numpy().tolist())

        if logits.ndim == 3:
            meter.sum.Lall = F.cross_entropy(logits.permute(0, 2, 1), ys)
        else:
            meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f'set seed {params.seed}')

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        super().on_hooked(source, params)
        self.accuracy = 0

    def on_eval_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_eval_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                self.pred = []
                self.true = []

    def on_test_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                self.pred = []
                self.true = []

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        if self.is_main:
            if params.get('confuse_matrix', False):
                from sklearn import metrics
                if len(self.pred) > 0:
                    cm = metrics.confusion_matrix(self.pred, self.true, labels=range(params.n_classes))
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
                        self.database.flush()

                    self.logger.info('Best Results', m)
                    self.pred_info.append([self.true, self.pred])
                    self.pred_info.flush()

    def save_best_model(self):
        file = self.exp.blob_file('best_model.ckpt', 'models')
        file_info = self.exp.blob_file('best_model.json', 'models')
        torch.save(self.state_dict(), file)
        with open(file_info, 'w') as w:
            w.write(json.dumps({'global_steps': self.global_steps, 'accuracy': self.accuracy}))
        self.logger.info(f'saved best model at {file}')

    def save_last_model(self):
        file = self.exp.blob_file('last_model.ckpt', 'models')
        file_info = self.exp.blob_file('last_model.json', 'models')
        torch.save(self.state_dict(), file)
        with open(file_info, 'w') as w:
            w.write(json.dumps({'global_steps': self.global_steps, 'accuracy': self.accuracy}))
        self.logger.info(f'saved last model at {file}')

    @classmethod
    def generate_exp_name(cls) -> str:
        return f'{cls.dirname()}.{cls.filebasename()}'


def main(
        trainer_cls: ClassVar[Trainer],
        params_cls: ClassVar[ParamsType],
        dm: ClassVar[DataModule],
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
