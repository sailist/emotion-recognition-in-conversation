"""
Reimplementation of paper COGMEN COntextualized GNN based Multimodal Emotion recognitioN

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

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch_geometric.nn import RGCNConv, TransformerConv

from contrib.make_optim import make_optim
from contrib.nn import TransformerEncoderLayer
from lumo import CollateBase
from lumo import Meter, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from .cogmen_utils import batch_graphify
from .mmbase import MMBaseTrainer, MMBaseParams, main


class COGMENParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.train.batch_size = 32
        self.val.batch_size = 32
        self.test.batch_size = 32

        self.num_heads = 17
        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 55
        self.optim = self.OPTIM.create_optim('Adam', lr=0.0001, weight_decay=1e-8)
        self.train.num_workers = 2
        self.test.num_workers = 2

    def iparams(self):
        super(COGMENParams, self).iparams()


ParamsType = COGMENParams


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, n_speakers=2):
        super(GNN, self).__init__()
        num_relations = 2 * n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=1, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, node_features, edge_index, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.relu(self.bn(self.conv2(x, edge_index)))

        return x


class COGMENModule(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_head,
                 n_speakers,
                 n_classes):
        super().__init__()

        find_head = False
        for h in range(6, num_head):
            if input_size % h == 0:
                num_head = h
                find_head = True
                break
        assert find_head, input_size

        encoder_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_head,
            dropout=0.5,
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )
        transformer_out = nn.Linear(
            input_size, hidden_size, bias=True
        )

        self.rnn = nn.ModuleList([
            transformer_encoder,
            transformer_out])

        # num_relations 其实就是每个 speaker 之间的联系，可以直接用 type embedding 来实现，一个 speaker 是一个 type
        # 另外一种解决思路是，每一类 relation 构建一个 transformer 层建模，也就是同 speaker 来一次，不同 speaker 来一次，所有 speaker 来一次

        self.gcn = GNN(hidden_size, hidden_size, hidden_size)

        self.cls = nn.Sequential(*[
            # MaskedEmotionAtt(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, n_classes),
        ])

        edge_type_to_idx = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                input_tensor,
                speaker_tensor,
                text_length,
                *args,
                **kwargs
                ):
        node_features = input_tensor
        for mod in self.rnn:
            node_features = mod(input_tensor)

        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            text_length,
            speaker_tensor,
            5,
            5,
            self.edge_type_to_idx,
        )

        graph_out = self.gcn(features, edge_index, edge_type)

        return self.cls(graph_out), features


class COGMENTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = COGMENModule(input_size=params.hidden_all,
                                  hidden_size=100,
                                  num_head=params.num_heads,
                                  n_speakers=params.n_speakers,
                                  n_classes=params.n_classes)
        self.optim = params.optim.build(self.model.parameters())
        self.logger.raw(self.optim)
        self.to_device()

    def to_logits(self, xs):
        return self.model(**xs)[0]

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']
        graph_out, features = self.model(**batch)
        Lall = F.cross_entropy(graph_out, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(graph_out.argmax(dim=-1), ys).float().mean()

        return meter


main = partial(main, COGMENTrainer, ParamsType)
