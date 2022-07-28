"""
Reimplementation of paper DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation
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
from .cogmen_utils import batch_graphify
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

        self.num_heads = 17
        self.gnn_heads = 1
        self.confuse_matrix = True
        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 55
        self.optim = self.OPTIM.create_optim('Adam', lr=0.0001, weight_decay=1e-8)
        self.split_params = False
        self.train.num_workers = 2
        self.test.num_workers = 2
        self.ema = True

        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None

    def iparams(self):
        super(DGCNParams, self).iparams()


ParamsType = DGCNParams


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


class DGCNModule(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_head,
                 n_speakers,
                 n_classes):
        super().__init__()
        gnn_nheads = 1

        find_head = False
        for h in range(6, num_head):
            if input_size % h == 0:
                num_head = h
                find_head = True
                break
        assert find_head

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


class DGCNTrainer(MMBaseTrainer):

    def icallbacks(self, params: ParamsType):
        super().icallbacks(params)
        callbacks.EvalCallback(eval_per_epoch=1, test_per_epoch=1).hook(self)
        callbacks.LoggerCallback(step_frequence=1, break_in=150).hook(self)
        callbacks.AutoLoadModel().hook(self)
        if isinstance(self, callbacks.BaseCallback):
            self.hook(self)

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = DGCNModule(input_size=params.hidden_all,
                                hidden_size=100,
                                num_head=params.num_heads,
                                n_speakers=params.n_speakers,
                                n_classes=params.n_classes)
        self.optim = make_optim(self.model, params.optim, split=params.split_params)
        self.logger.raw(self.optim)
        self.to_device()

        self.lr_sche = lr_scheduler.ReduceLROnPlateau(self.optim, "min")

        if params.ema:
            self.ema_model = EMA(self.model, alpha=0.999)

    def to_logits(self, xs):
        return self.model(**xs)[0]

    def to_ema_logits(self, xs):
        return self.ema_model(**xs)[0]

    def train_step(self, batch, params: ParamsType = None) -> MetricType:
        super().train_step(batch, params)
        meter = Meter()

        ys = batch['label']
        graph_out, features = self.model(**batch)
        Lall = F.cross_entropy(graph_out, ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        self.optim.step()

        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(graph_out.argmax(dim=-1), ys).float().mean()

        return meter


class DGCNCollate(CollateBase):
    def __init__(self, params: ParamsType):
        super().__init__(params)
        self.modalities = params.modality
        self.speaker_to_idx = {"M": 0, "F": 1}

    def __call__(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s['text']) for s, in samples]).long()
        mx = torch.max(text_len_tensor).item()

        input_tensor = []
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        labels = []
        utterances = []
        for i, (dic,) in enumerate(samples):
            sentence = dic['sentence']
            speakers = dic['speakers']

            visual = dic['visual']
            audio = dic['audio']
            textual = dic['text']
            label = dic['label']

            cur_len = len(textual)
            utterances.append(sentence)
            tmp = []
            for t, a, v in zip(textual, audio, visual):
                res = {'t': t, 'a': a, 'v': v}
                nres = torch.cat([torch.from_numpy(res[i]) for i in self.modalities])
                tmp.append(nres)

            tmp = torch.stack(tmp)
            tmp = torch.cat([tmp, torch.zeros(mx - len(tmp), tmp.shape[1], dtype=tmp.dtype)])
            input_tensor.append(tmp)

            speaker_tensor[i, :cur_len] = torch.tensor(speakers).argmax(dim=-1)

            labels.extend(label)

        label = torch.tensor(labels).long()
        data = {
            "text_length": text_len_tensor,
            "input_tensor": torch.stack(input_tensor),
            "speaker_tensor": speaker_tensor,
            "label": label,
            "utterance_texts": utterances,
        }
        return data


class DGCNDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = DGCNCollate(params)
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


main = partial(main, DGCNTrainer, ParamsType, DGCNDM)
