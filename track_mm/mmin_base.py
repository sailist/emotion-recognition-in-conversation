"""
reimplementation of paper MMINBase COntextualized GNN based Multimodal Emotion recognitioN
"""
from typing import ClassVar
import json
import torch
from torch.optim import lr_scheduler

from contrib.make_optim import make_optim
from lumo import Trainer, TrainerParams, Meter, callbacks, DataModule, Record, MetricType, TrainStage
from lumo.contrib import EMA
from models.module_utils import ModelParams
from mmdatasets.dataset_utils import DataParams
import numpy as np
from mmdatasets.mmin_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from lumo import CollateBase

from torch.nn import functional as F
from .mmin_models import MMINBaseModule

from torch.nn.utils.rnn import pad_sequence


class MMINBaseParams(TrainerParams, ModelParams, DataParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.train.batch_size = 32
        self.val.batch_size = 32
        self.test.batch_size = 32

        self.num_heads = 10
        self.gnn_heads = 1
        self.confuse_matrix = True
        self.dataset = 'iemocap-mmin-4'
        self.epoch = 55
        self.optim = self.OPTIM.create_optim('Adam', lr=0.0002, weight_decay=0)
        self.split_params = False
        self.train.num_workers = 2
        self.test.num_workers = 2
        self.ema = True

        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None

    def iparams(self):
        super(MMINBaseParams, self).iparams()
        if self.get('debug'):
            self.train.batch_size = 2
            self.test.batch_size = 2
            self.train.num_workers = 0
            self.test.num_workers = 0


ParamsType = MMINBaseParams


class MMINBaseTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

    def on_process_loader_begin(self, trainer: 'Trainer', func, params: ParamsType, dm: DataModule, stage: TrainStage,
                                *args, **kwargs):
        super().on_process_loader_begin(trainer, func, params, dm, stage, *args, **kwargs)
        if stage.is_train():
            self.rnd.mark(params.seed)
            self.logger.info(f'set seed {params.seed}')

    def on_hooked(self, source: 'Trainer', params: ParamsType):
        super().on_hooked(source, params)
        self.accuracy = 0

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        input_size = 1380

        self.model = MMINBaseModule(visual_dim=342,
                                    text_dim=1024,
                                    audio_dim=130,
                                    n_classes=params.n_classes)
        self.optim = params.optim.build(self.model.parameters())
        self.logger.raw(self.optim)
        self.to_device()

        self.lr_sche = lr_scheduler.ReduceLROnPlateau(self.optim, "min")

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def to_logits(self, xs):
        return self.model(**xs)[0]

    def to_ema_logits(self, xs):
        return self.ema_model(**xs)[0]

    def test_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if params.confuse_matrix:
            self.true.extend(ys.cpu().numpy().tolist())
            self.pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']
        logits, features = self.model(**batch)
        Lall = F.cross_entropy(logits, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(logits.argmax(dim=-1), ys).float().mean()

        return meter

    def evaluate_step(self, batch, params: ParamsType = None) -> MetricType:
        meter = Meter()
        ys = batch['label']
        logits = self.to_logits(batch)

        if params.confuse_matrix:
            self.true.extend(ys.cpu().numpy().tolist())
            self.pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        meter.sum.Lall = F.cross_entropy(logits, ys)

        meter.sum.Acc = torch.eq(logits.argmax(dim=-1), ys).sum()
        if params.ema:
            logits2 = self.to_ema_logits(batch)
            meter.sum.Acc2 = torch.eq(logits2.argmax(dim=-1), ys).sum()
        meter.sum.C = ys.shape[0]
        return meter

    def on_eval_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_eval_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.confuse_matrix:
                self.pred = []
                self.true = []

    def on_eval_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_eval_end(trainer, func, params, record, *args, **kwargs)
        if self.is_main:
            l = record.agg()['Lall']
            self.logger.info(f'Evalute Loss: {l}')
            self.lr_sche.step(record.agg()['Lall'])
            self.logger.raw(self.optim)

    def on_test_begin(self, trainer: 'Trainer', func, params: ParamsType, *args, **kwargs):
        super().on_test_begin(trainer, func, params, *args, **kwargs)
        if self.is_main:
            if params.confuse_matrix:
                self.pred = []
                self.true = []

    def on_test_end(self, trainer: 'Trainer', func, params: ParamsType, record: Record = None, *args, **kwargs):
        super().on_test_end(trainer, func, params, record, *args, **kwargs)
        if self.is_main:
            if params.confuse_matrix:
                from sklearn import metrics
                if len(self.pred) > 0:
                    self.logger.raw(
                        metrics.classification_report(
                            self.true, self.pred, target_names=['hap', 'sad', 'neu', 'ang'], digits=4
                        )
                    )

            if self.accuracy < record.agg()['Acc']:
                self.accuracy = record.agg()['Acc']
                self.save_best_model()
            self.save_last_model()

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


class MMINBaseCollate(CollateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = 'atv'
        self.speaker_to_idx = {"M": 0, "F": 1}

    def __call__(self, samples):

        data = {}
        for key in ['audio_feature', ]:
            features = [torch.from_numpy(sample_dic[key]).float() for (sample_dic,) in samples]
            pad_features = pad_sequence(features, batch_first=True, padding_value=0)
            data[key] = pad_features
            data[f'{key}_length'] = [len(feat) for feat in features]

        for key in ['visual_feature',
                    'text_feature']:
            features = [sample_dic[key] for (sample_dic,) in samples]
            pad_features = np.stack(features)
            data[key] = torch.from_numpy(pad_features).float()

        label = [sample_dic['label'] for (sample_dic,) in samples]
        name = [sample_dic['name'] for (sample_dic,) in samples]

        data['label'] = torch.tensor(label, dtype=torch.long)
        data['name'] = name

        return data


class MMINBaseDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = MMINBaseCollate()
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   method=params.get('method'))
            dl = ds.DataLoader(**params.train.to_dict(), collate_fn=collate_fn)
        elif stage.is_val():
            ds = get_val_dataset(params.dataset,
                                 method=params.get('method'))
            dl = ds.DataLoader(**params.val.to_dict(), collate_fn=collate_fn)
        else:
            ds = get_test_dataset(params.dataset,
                                  method=params.get('method'))
            dl = ds.DataLoader(**params.test.to_dict(), collate_fn=collate_fn)
        print(ds, stage)

        self.regist_dataloader_with_stage(stage, dl)


def main(
        trainer_cls: ClassVar[Trainer] = MMINBaseTrainer,
        params_cls: ClassVar[ParamsType] = ParamsType,
        dm: ClassVar[DataModule] = MMINBaseDM,
):
    params = params_cls()
    params.from_args()

    dm = dm(params)
    trainer = trainer_cls(params, dm)

    if params.pretrain_path is not None and params.train_linear:
        trainer.load_state_dict(params.pretrain_path)
        trainer.test()
        return

    if params.get('eval_first', False):
        trainer.test()

    trainer.rnd.mark(params.seed)
    trainer.train()
    trainer.save_model()
