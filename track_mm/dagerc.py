"""
DAG-ERC
Reimplementation of Directed Acyclic Graph Network for Conversational Emotion Recognition
"""
from torch.optim import lr_scheduler

from contrib.make_optim import make_optim
from lumo import Meter, callbacks, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from lumo import CollateBase

from functools import partial

from .dagerc_models import *
# from .dagerc_utils import *
from .mmbase import MMBaseTrainer, MMBaseParams, main


class DAGERCParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.epoch = 20
        self.train.batch_size = 8
        self.test.batch_size = 8

        self.num_heads = 10
        self.gnn_heads = 1
        self.gnn_layers = 2
        self.dropout = 0
        self.confuse_matrix = True
        self.dataset = 'iemocap-cogmen-6'
        self.epoch = 30
        self.optim = self.OPTIM.create_optim('AdamW', lr=1e-3)
        self.split_params = False
        self.train.num_workers = 2
        self.test.num_workers = 2
        self.ema = True

        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None

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

    def forward(self, input_tensor, adj, s_mask, text_length, **kwargs):
        num_utter = input_tensor.size()[1]

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
        self.optim = make_optim(self.model, params.optim, split=params.split_params)
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
        attention_mask = (ys >= 0)
        graph_out, features = self.model(**batch)
        Lall = F.cross_entropy(graph_out.permute(0, 2, 1), ys)

        self.optim.zero_grad()
        self.accelerate.backward(Lall)
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optim.step()

        if params.ema:
            self.ema_model.step()

        with torch.no_grad():
            meter.Lall = Lall
            meter.Acc = torch.eq(graph_out[attention_mask].argmax(dim=-1), ys[attention_mask]).float().mean()

        return meter


class DAGERCCollate(CollateBase):
    def __init__(self, params: ParamsType):
        super().__init__(params)
        self.modalities = params.modality
        self.windowp = 1

    def get_adj_v1(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
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
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
         s_mask_onehot (B, N, N, 2) onehot emcoding of s_mask
        '''
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

    def __call__(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s['text']) for s, in samples]).long()
        mx = torch.max(text_len_tensor).item()

        input_tensor = []
        speaker_tensor = []
        # speaker_tensor = torch.zeros((batch_size, mx)).long()
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

            speaker_tensor.append(speakers)

            labels.append(label + [-100] * (mx - len(label)))

        adj = self.get_adj_v1(speaker_tensor, mx)
        s_mask, s_mask_onehot = self.get_s_mask(speaker_tensor, mx)
        label = torch.tensor(labels).long()
        data = {
            "text_length": text_len_tensor,
            "input_tensor": torch.stack(input_tensor),
            "adj": adj,
            "s_mask": s_mask,
            "label": label,
            "utterance_texts": utterances,
        }
        return data


class DAGERCDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = DAGERCCollate(params)
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

        self.regist_dataloader_with_stage(stage, dl)


main = partial(main, DAGERCTrainer, ParamsType, DAGERCDM)
