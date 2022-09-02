"""
reimplementation of paper MMINMiss COntextualized GNN based Multimodal Emotion recognitioN

/home/admin/.lumo/experiment/track_mm.mmin_base/220707.000.7dt/l.0.2207071001.log
"""
import random
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
from models.init import efficiency_init
from torch import nn
from torch.nn import functional as F
from .mmin_models import TextCNN, LSTMEncoder, Classifier, ResidualAE, MMINBaseModule
from itertools import chain
from torch.nn.utils.rnn import pad_sequence


class MMINMissParams(TrainerParams, ModelParams, DataParams):

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
        self.pretrain_path = '/home/admin/.lumo/blob/track_mm.mmin_base/220707.001.0ft/models/best_model.ckpt'
        self.finetune = False

    def iparams(self):
        super(MMINMissParams, self).iparams()
        if self.get('debug'):
            self.train.batch_size = 2
            self.test.batch_size = 2
            self.train.num_workers = 0
            self.test.num_workers = 0


ParamsType = MMINMissParams


class MMINMissModule(nn.Module):
    def __init__(self, visual_dim=0, text_dim=0, audio_dim=0, n_classes=4):
        super().__init__()

        input_dim = 128 * 3
        self.netL = TextCNN(text_dim, 128)
        self.netA = LSTMEncoder(audio_dim, 128, embd_method='maxpool')
        self.netV = LSTMEncoder(visual_dim, 128, embd_method='maxpool')

        ae_layers = [256, 128, 64]
        n_blocks = 5
        self.netAE = ResidualAE(ae_layers, n_blocks, 384, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(ae_layers, n_blocks, 384, dropout=0, use_bn=False)

        self.netC = Classifier(ae_layers[-1] * n_blocks, [128, 128], n_classes, dropout=0.3, use_bn=False)

    def forward(self,
                audio_feature=None,
                visual_feature=None,
                text_feature=None,
                *args,
                **kwargs
                ):
        features = []
        if audio_feature is not None:
            audio_feature = self.netA(audio_feature)
            features.append(audio_feature)
        if visual_feature is not None:
            visual_feature = self.netV(visual_feature)
            features.append(visual_feature)
        if text_feature is not None:
            text_feature = self.netL(text_feature)
            features.append(text_feature)

        features = torch.cat(features, dim=-1)
        fusion, latent = self.netAE(features)
        fusion_cycle, _ = self.netAE_cycle(features)

        logits, _ = self.netC(latent)
        return logits, fusion, fusion_cycle, features


class MMINMissTrainer(Trainer, callbacks.TrainCallback, callbacks.InitialCallback):

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

        self.modelB = MMINBaseModule(visual_dim=342,
                                     text_dim=1024,
                                     audio_dim=130,
                                     n_classes=params.n_classes)
        if params.pretrain_path is not None:
            sd = torch.load(params.pretrain_path, map_location='cpu')
            self.modelB.load_state_dict(sd['models']['model'])
            self.logger.info('load pretrained path')

        self.optim = params.optim.build(chain(*[
            self.model.parameters(),
            self.modelB.parameters()
        ]))
        self.logger.raw(self.optim)

        self.model.apply(efficiency_init)
        self.modelB.apply(efficiency_init)

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
        logits, fusion_feature = self.model.forward(**batch)

        reverse_logits, reverse_features = self.modelB.forward(
            **{'audio_feature': batch['audio_feature_reverse'],
               'visual_feature': batch['visual_feature_reverse'],
               'text_feature': batch['text_feature_reverse']}
        )

        # reverse_features = torch.cat(reverse_features, dim=-1)

        Lce = F.cross_entropy(logits, ys)
        Lrce = F.cross_entropy(reverse_logits, ys)
        Lmse = F.mse_loss(reverse_features, fusion_feature)

        Lall = Lce + Lmse * 4 + Lrce

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


class MMINMissCollate(CollateBase):
    def __init__(self, has_miss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = 'atv'
        self.speaker_to_idx = {"M": 0, "F": 1}
        self.has_miss = has_miss

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

        if self.has_miss:
            missing_type = [sample_dic['missing_type'] for (sample_dic,) in samples]

            missing_type = torch.tensor(missing_type, dtype=torch.long)
            for i, key in enumerate(['visual_feature', 'text_feature', 'audio_feature']):
                features = data[key]
                missing_index = missing_type[:, i][:, None, None]
                data[key] = features * missing_index
                data[f'{key}_reverse'] = features * -1 * (missing_index - 1)
            data['missing_type'] = missing_type

        data['label'] = torch.tensor(label, dtype=torch.long)
        data['name'] = name

        return data


class Missing:

    def __init__(self):
        self.missing_type = [[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 1, 0],
                             [1, 0, 1],
                             [0, 1, 1],
                             ]
        self.modal_keys = ['visual_feature', 'text_feature', 'audio_feature']
        self.missing_category = 'zero'

    def __call__(self, dic):
        typ = random.choice(self.missing_type)
        dic['missing_type'] = typ
        return dic


class MMINMissDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = MMINMissCollate(has_miss=stage.is_train())
        if stage.is_train():
            ds = get_train_dataset(params.dataset,
                                   method=params.get('method'))
            ds.add_output_transform('all', Missing())
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
        trainer_cls: ClassVar[Trainer] = MMINMissTrainer,
        params_cls: ClassVar[ParamsType] = ParamsType,
        dm: ClassVar[DataModule] = MMINMissDM,
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
