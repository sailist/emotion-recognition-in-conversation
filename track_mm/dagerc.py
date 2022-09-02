"""
DAG-ERC
Reimplementation of Directed Acyclic Graph Network for Conversational Emotion Recognition
"""
from functools import partial

import torch
from torch.optim import lr_scheduler

from contrib.make_optim import make_optim
from lumo import CollateBase
from lumo import Meter, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from .dagerc_models import *
# from .dagerc_utils import *
from .mmbase import MMBaseTrainer, MMBaseParams, main


class DAGERCParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.train.batch_size = 8
        self.test.batch_size = 8

        self.num_heads = 10
        self.gnn_heads = 1
        self.gnn_layers = 4
        self.dropout = 0

        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 30

        self.optim = self.OPTIM.create_optim('AdamW', lr=1e-3)

        self.speaker_onehot = True

    def iparams(self):
        super(DAGERCParams, self).iparams()
        if self.reimplement:
            if 'iemocap' in self.dataset:
                self.dropout = 0.2
                self.epoch = 55
                self.train.batch_size = 16
                self.optim.lr = 0.0005
                self.gnn_layers = 4
            elif 'meld' in self.dataset:
                self.optim.lr = 0.00001
                self.train.batch_size = 64
                self.epoch = 70
                self.dropout = 0.1
            elif 'emorynlp' in self.dataset:
                self.optim.lr = 0.00005
                self.train.batch_size = 32
                self.epoch = 100
                self.dropout = 0.3
            elif 'dailydialog' in self.dataset:
                self.gnn_layers = 3
                self.optim.lr = 0.00002
                self.train.batch_size = 64
                self.epoch = 50
                self.dropout = 0.3


ParamsType = DAGERCParams


class DAGERCModule(nn.Module):
    def __init__(self,
                 emb_dim=100,
                 dropout=0.2,
                 n_classes=7,
                 gnn_layers=4,
                 ):
        super().__init__()

        self.rel_attn = True
        self.nodal_att_type = None
        self.dropout = nn.Dropout(dropout)

        hidden_dim = 300

        self.gnn_layers = gnn_layers
        self.gather = nn.ModuleList([GAT_dialoggcn_v1(hidden_dim) for i in range(gnn_layers)])
        self.grus_c = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for i in range(gnn_layers)])
        self.grus_p = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for i in range(gnn_layers)])

        self.fcs = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for i in range(gnn_layers)])
        self.fc1 = nn.Linear(emb_dim, hidden_dim)

        self.windowp = 1

        in_dim = hidden_dim * (gnn_layers + 1) + emb_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.attentive_node_features = attentive_node_features(in_dim)

    def get_adj_v1(self, speakers, max_dialog_len):
        """
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        """
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == self.windowp:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        """
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        """
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1

            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def forward(self, input_tensor, text_length, speaker_tensor, **kwargs):
        num_utter = input_tensor.size()[1]
        speaker_tensor = speaker_tensor.tolist()
        mx = torch.max(text_length).item()
        adj = self.get_adj_v1(speaker_tensor, mx).to(input_tensor.device)
        s_mask, _ = self.get_s_mask(speaker_tensor, mx)
        s_mask = s_mask.to(input_tensor.device)
        # s_mask_onehot = s_mask_onehot.to(input_tensor.device)
        H0 = F.relu(self.fc1(input_tensor))
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.gnn_layers):
            C = self.grus_c[l](H[l][:, 0, :]).unsqueeze(1)
            M = torch.zeros_like(C).squeeze(1)
            # P = M.unsqueeze(1)
            P = self.grus_p[l](M, H[l][:, 0, :]).unsqueeze(1)
            # H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))
            # H1 = F.relu(C+P)
            H1 = C + P
            for i in range(1, num_utter):
                # print(i,num_utter)
                _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], s_mask[:, i, :i])
                # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])

                C = self.grus_c[l](H[l][:, i, :], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:, i, :]).unsqueeze(1)
                # P = M.unsqueeze(1)
                # H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))
                # H_temp = F.relu(C+P)
                H_temp = C + P
                H1 = torch.cat((H1, H_temp), dim=1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(input_tensor)

        H = torch.cat(H, dim=2)

        H = self.attentive_node_features(H, text_length, self.nodal_att_type)

        logits = self.out_mlp(H)

        return logits, None


class DAGERCTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = DAGERCModule(emb_dim=params.hidden_all,
                                  dropout=params.dropout,
                                  gnn_layers=params.gnn_layers,
                                  n_classes=params.n_classes)
        self.optim = params.optim.build(self.model.parameters())
        self.to_device()

        self.lr_sche = lr_scheduler.ReduceLROnPlateau(self.optim, "min")

    def to_logits(self, xs):
        return self.model(**xs)[0]

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']

        attention_mask = batch['attention_mask']
        graph_out, features = self.model(**batch)
        logits = graph_out[attention_mask.bool()]
        Lall = F.cross_entropy(logits, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optim.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(logits.argmax(dim=-1), ys).float().mean()

        return meter


main = partial(main, DAGERCTrainer, ParamsType)
