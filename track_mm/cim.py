"""
Reimplementation of paper CIM COntextualized GNN based Multimodal Emotion recognitioN

pre: 0.8171300823486798 | cls_pre: [0.81896552 0.88085106 0.81481481 0.72897196] |
rec: 0.8123011664899258 | cls_rec: [0.65972222 0.84489796 0.80208333 0.91764706] |
f1: 0.8113398367774792 | cls_f1: [0.73076923 0.8625     0.80839895 0.8125    ] |
acc: 0.8123011664899258 |
wa: 0.8060876433906896 | mif1: 0.8123011664899258 | maf1: 0.8035420452251161

pre: 0.6571254765429705 | cls_pre: [0.83333333 0.79812207 0.61096606 0.53112033 0.67630058 0.5875576 ] |
rec: 0.6327788046826864 | cls_rec: [0.04861111 0.69795918 0.62239583 0.75294118 0.77926421 0.65354331] |
f1: 0.6111887097722818 | cls_f1: [0.08917197 0.75496689 0.61518662 0.64974619 0.72360248 0.60657734] |
acc: 0.6327788046826864 |
wa: 0.5931933134919932 | mif1: 0.6327788046826864 | maf1: 0.579364226239531

"""
from functools import partial

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.modeling_bert import BertAttention
import torch
from torch import nn
from torch.nn import functional as F
from lumo import Meter, MetricType
from .mmbase import MMBaseTrainer, MMBaseParams, main


class CIMParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.train.batch_size = 16
        self.val.batch_size = 32
        self.test.batch_size = 32

        self.num_heads = 17
        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 55
        self.optim = self.OPTIM.create_optim('Adam', lr=0.001)
        self.train.num_workers = 2
        self.test.num_workers = 2

        self.apply_multi = True
        self.apply_bin = True

        self.metric = 'multiemo'

    def iparams(self):
        super(CIMParams, self).iparams()
        if 'mosei' not in self.dataset:
            self.apply_multi = False

        if self.n_classes != 2:
            self.mosei_metric = ''


ParamsType = CIMParams


class CIMModule(nn.Module):
    def __init__(self,
                 text_dim,
                 audio_dim,
                 visual_dim,
                 hidden_size,
                 n_classes,
                 drop0=0.3,
                 drop1=0.3,
                 ):
        super().__init__()
        self.rnn = nn.ModuleDict({
            't': nn.GRU(text_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True),
            'a': nn.GRU(audio_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True),
            'v': nn.GRU(visual_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True),
        })

        self.rnn_adapter = nn.ModuleDict({
            't': nn.Linear(text_dim, hidden_size * 2),
            'a': nn.Linear(audio_dim, hidden_size * 2),
            'v': nn.Linear(visual_dim, hidden_size * 2),
        })

        self.drop0 = nn.ModuleDict({
            't': nn.Dropout(drop0),
            'a': nn.Dropout(drop0),
            'v': nn.Dropout(drop0),
        })

        self.adapter = nn.ModuleDict({
            't': nn.Sequential(nn.Linear(hidden_size * 2, 100), nn.ReLU()),
            'a': nn.Sequential(nn.Linear(hidden_size * 2, 100), nn.ReLU()),
            'v': nn.Sequential(nn.Linear(hidden_size * 2, 100), nn.ReLU()),
        })

        self.drop1 = nn.ModuleDict({
            't': nn.Dropout(drop1),
            'a': nn.Dropout(drop1),
            'v': nn.Dropout(drop1),
        })

        self.cls2 = nn.Linear(100 * 9, n_classes)
        self.cls7 = nn.Linear(100 * 9, 7)

    def attention_op(self, x, y, attention_mask, type='simple'):
        m_dash = torch.matmul(x, y.transpose(-1, -2))
        extended_attention_mask = (1.0 - attention_mask[:, None, :]) * -10000.0
        m_dash = m_dash + extended_attention_mask

        m = torch.softmax(m_dash, dim=-1)
        h_dash = torch.matmul(m, y)
        return h_dash * x

    def forward(self,
                text_feature=None, audio_feature=None, visual_feature=None,
                text_length=None, attention_mask=None,
                *args,
                **kwargs
                ):
        # rnn_audio = self.rnn_adapter['a'](audio_feature)
        # rnn_visual = self.rnn_adapter['v'](visual_feature)
        # rnn_text = self.rnn_adapter['t'](text_feature)

        packed_audio_feature = pack_padded_sequence(audio_feature, text_length.cpu(),
                                                    batch_first=True, enforce_sorted=False)
        packed_visual_feature = pack_padded_sequence(visual_feature, text_length.cpu(),
                                                     batch_first=True, enforce_sorted=False)
        packed_text_feature = pack_padded_sequence(text_feature, text_length.cpu(),
                                                   batch_first=True, enforce_sorted=False)

        rnn_audio, (_, _) = self.rnn['a'](packed_audio_feature)
        rnn_visual, (_, _) = self.rnn['v'](packed_visual_feature)
        rnn_text, (_, _) = self.rnn['t'](packed_text_feature)

        rnn_audio, _ = pad_packed_sequence(rnn_audio, batch_first=True)
        rnn_visual, _ = pad_packed_sequence(rnn_visual, batch_first=True)
        rnn_text, _ = pad_packed_sequence(rnn_text, batch_first=True)

        rnn_audio = self.drop0['a'](rnn_audio)
        rnn_visual = self.drop0['v'](rnn_visual)
        rnn_text = self.drop0['t'](rnn_text)

        dense_audio = self.adapter['a'](rnn_audio)
        dense_visual = self.adapter['v'](rnn_visual)
        dense_text = self.adapter['t'](rnn_text)

        dense_audio = self.drop1['a'](dense_audio)
        dense_visual = self.drop1['v'](dense_visual)
        dense_text = self.drop1['t'](dense_text)

        av = self.attention_op(dense_audio, dense_visual, attention_mask)
        at = self.attention_op(dense_audio, dense_text, attention_mask)
        va = self.attention_op(dense_visual, dense_audio, attention_mask)
        vt = self.attention_op(dense_visual, dense_text, attention_mask)
        ta = self.attention_op(dense_text, dense_audio, attention_mask)
        tv = self.attention_op(dense_text, dense_visual, attention_mask)

        merged = torch.cat([
            av,
            va,
            ta,
            tv,
            at,
            vt,
            dense_audio, dense_visual, dense_text], dim=-1)

        logits1 = self.cls2(merged)
        logits2 = self.cls7(merged)

        return logits1[attention_mask.bool()], logits2[attention_mask.bool()]


class CIMTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = CIMModule(text_dim=params.hidden_text,
                               audio_dim=params.hidden_audio,
                               visual_dim=params.hidden_visual,
                               hidden_size=200,
                               n_classes=params.n_classes,
                               )
        self.optim = params.optim.build(self.model.parameters())
        self.logger.raw(self.optim)
        self.to_device()

    def to_mosei_multitask_logits(self, xs):
        logits2, logits7 = self.model(**xs)
        return logits2, logits7

    def to_logits(self, xs):
        logits2, logits7 = self.model(**xs)
        return logits2

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']
        logits2, logits7 = self.model(**batch)

        Lce = F.cross_entropy(logits2, ys)

        Lmulti = 0
        if params.apply_multi:
            Lmulti = F.binary_cross_entropy_with_logits(logits7, batch['emo_label'].float())

        Lall = 0
        if params.apply_bin:
            Lall = Lall + Lce

        if params.apply_multi:
            Lall = Lall + Lmulti

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Lce = Lce
            meter.Lmulti = Lmulti

            meter.Acc = torch.eq(logits2.argmax(dim=-1), ys).float().mean()

        return meter


main = partial(main, CIMTrainer, ParamsType)
