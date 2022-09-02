"""
Reimplementation of paper DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation

Adapted from https://github.com/declare-lab/conv-emotion
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from lumo import CollateBase
from lumo.contrib.torch.tensor import onehot
from lumo import Meter, callbacks, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from .dgcnv2_models import (batch_graphify, DialogueRNN, pad_sequence, MaskedEdgeAttention, GraphNetwork)

from .mmbase import MMBaseTrainer, MMBaseParams, main


class DGCNParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.module = None
        self.method = None

        self.train.batch_size = 32
        self.val.batch_size = 32
        self.test.batch_size = 32

        self.base_model = self.choice('LSTM', 'DialogRNN', 'GRU', 'None')

        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 55

        self.optim = self.OPTIM.create_optim('Adam', lr=0.0003, weight_decay=0)

        self.train.num_workers = 2
        self.test.num_workers = 2

        self.loss_weights = True
        self.speaker_onehot = True
        self.batch_first = False

    def iparams(self):
        super(DGCNParams, self).iparams()


ParamsType = DGCNParams


class DGCNModule(nn.Module):

    def __init__(self, base_model,
                 input_size=100,
                 hidden_size=100,
                 n_speakers=2,
                 window_past=10, window_future=10,
                 n_classes=7,
                 listener_state=False, context_attention='general', dropout_rec=0.5, dropout=0.4,
                 nodal_attention=True, avec=False):

        super(DGCNModule, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.n_speakers = n_speakers

        D_g = 150
        D_p = 150

        D_h = 100
        D_a = 100
        graph_hidden_size = 100
        max_seq_len = 110

        # The base model is the sequential context encoder.
        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(input_size, D_g, D_p, hidden_size, listener_state, context_attention, D_a,
                                            dropout_rec)
            self.dialog_rnn_r = DialogueRNN(input_size, D_g, D_p, hidden_size, listener_state, context_attention, D_a,
                                            dropout_rec)

        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                                dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True,
                              dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(input_size, 2 * hidden_size)

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * hidden_size, max_seq_len)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2 * hidden_size, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, input_tensor, speaker_tensor, attention_mask, text_length, **kwargs):
        """

        text_length
        input_tensor
        speaker_tensor

        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        # speaker_tensor = onehot(speaker_tensor, self.n_speakers).transpose(0, 1)  # batch_second

        if self.base_model == "DialogRNN":

            if self.avec:
                emotions, _ = self.dialog_rnn_f(input_tensor, speaker_tensor)

            else:
                emotions_f, alpha_f = self.dialog_rnn_f(input_tensor, speaker_tensor)  # seq_len, batch, D_e
                rev_U = self._reverse_seq(input_tensor, attention_mask)
                rev_qmask = self._reverse_seq(speaker_tensor, attention_mask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, attention_mask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(input_tensor)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(input_tensor)

        elif self.base_model == 'None':
            emotions = self.base_linear(input_tensor)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, speaker_tensor,
                                                                                        text_length,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model)
        logits = self.graph_net(features, edge_index, edge_norm, edge_type, text_length, attention_mask,
                                self.nodal_attention,
                                self.avec)

        flattened_logits = logits.transpose(0, 1)[attention_mask.bool()]

        return flattened_logits, features


class DGCNTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = DGCNModule(
            base_model=params.base_model,
            input_size=params.hidden_all,
            hidden_size=100,
            n_speakers=params.n_speakers,
            n_classes=params.n_classes,
            context_attention='general',
        )

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
