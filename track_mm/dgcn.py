"""
Reimplementation of paper DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation

Adapted from https://github.com/mianzhang/dialogue_gcn
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch_geometric.nn import RGCNConv, TransformerConv

from contrib.make_optim import make_optim
from contrib.nn import TransformerEncoderLayer
from lumo import CollateBase
from lumo import Meter, callbacks, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from .dgcn_models import (batch_graphify, SeqContext, Classifier, GCN, GraphConv, RGCNConv, EdgeAtt)

from .mmbase import MMBaseTrainer, MMBaseParams, main


class DGCNParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.train.batch_size = 32
        self.val.batch_size = 32
        self.test.batch_size = 32

        self.loss_weights = True

        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 55
        self.optim = self.OPTIM.create_optim('Adam', lr=0.0003, weight_decay=0)

        self.train.num_workers = 2
        self.test.num_workers = 2

    def iparams(self):
        super(DGCNParams, self).iparams()


ParamsType = DGCNParams


class DGCNModule(nn.Module):

    def __init__(self,
                 n_speakers,
                 input_size=100,
                 hidden_size=200,
                 context=[10, 10], dropout=0.4, n_classes=4):
        super(DGCNModule, self).__init__()
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100

        self.wp, self.wf = context

        self.rnn = SeqContext(input_size, hidden_size, dropout)
        self.edge_att = EdgeAtt(hidden_size, self.wp, self.wf)
        self.gcn = GCN(hidden_size, h1_dim, h2_dim, n_speakers=n_speakers)
        self.clf = Classifier(hidden_size + h2_dim, hc_dim, n_classes, dropout=dropout)

        edge_type_to_idx = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx

    def forward(self,
                input_tensor,
                speaker_tensor,
                text_length,
                **kwargs):

        node_features = self.rnn.forward(text_length, input_tensor)
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, text_length, speaker_tensor, self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att)
        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        logits = self.clf(torch.cat([features, graph_out], dim=-1), text_length)

        return logits, graph_out


class DGCNTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = DGCNModule(input_size=params.hidden_all,
                                hidden_size=200,
                                n_speakers=params.n_speakers,
                                n_classes=params.n_classes)
        self.optim = params.optim.build(self.model.parameters())
        self.logger.raw(self.optim)
        self.to_device()

        if params.loss_weights:
            self.loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                              1 / 0.160585, 1 / 0.127711, 1 / 0.252668], device=self.device)
        else:
            self.loss_weights = None

    def to_logits(self, xs):
        return self.model(**xs)[0]

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']
        graph_out, features = self.model(**batch)

        Lall = F.cross_entropy(graph_out, ys, weight=self.loss_weights)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(graph_out.argmax(dim=-1), ys).float().mean()

        return meter


main = partial(main, DGCNTrainer, ParamsType)
