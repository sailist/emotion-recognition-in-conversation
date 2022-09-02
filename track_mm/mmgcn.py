"""
Reimplementation of MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation


"""
from functools import partial

from torch.optim import lr_scheduler

from contrib.make_optim import make_optim
from lumo import CollateBase
from lumo import Meter, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from .mmbase import MMBaseTrainer, MMBaseParams, main
from .mmgcn_models import *
from .mmgcn_utils import *


class MMGCNParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.epoch = 60
        self.train.batch_size = 16
        self.test.batch_size = 16

        self.dataset = 'iemocap-cogmen-6'

        self.optim = self.OPTIM.create_optim('Adam', lr=0.0003, weight_decay=3e-5)

        self.train.num_workers = 2
        self.test.num_workers = 2

        self.speaker_onehot = True
        self.batch_first = False

    def iparams(self):
        super(MMGCNParams, self).iparams()
        if self.reimplement:
            if 'iemocap' in self.dataset:
                self.optim.lr = 0.0003
                self.optim.weight_decay = 3e-5
            elif 'meld' in self.dataset:
                self.optim.lr = 0.0001
                self.optim.weight_decay = 0


ParamsType = MMGCNParams


class MMGCNModule(nn.Module):
    def __init__(self,
                 hidden_text=100,
                 D_e=100,
                 graph_hidden_size=200, n_speakers=2, max_seq_len=200,
                 window_past=10, window_future=10,
                 n_classes=7,
                 nodal_attention=True,
                 hidden_visual=512, hidden_audio=100, modals='atv'):
        super().__init__()

        self.modals = modals
        self.linear_l = nn.Linear(hidden_text, 200)
        self.lstm_l = nn.LSTM(200, 100, 2, bidirectional=True, dropout=0.4)
        self.linear_a = nn.Linear(hidden_audio, 200)
        self.linear_v = nn.Linear(hidden_visual, 200)

        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len)
        self.nodal_attention = nodal_attention

        self.graph_model = MMGCN(a_dim=2 * D_e, v_dim=2 * D_e, l_dim=2 * D_e, n_dim=2 * D_e, nlayers=64,
                                 nhidden=graph_hidden_size, nclass=n_classes, dropout=0.4, lamda=0.5,
                                 alpha=0.1, variant=True, return_feature=True,
                                 use_residue=True, n_speakers=n_speakers, modals=self.modals,
                                 use_speaker=True, use_modal=False)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

        self.gatedatt = MMGatedAttention(2 * D_e + graph_hidden_size, graph_hidden_size, att_type='general')
        self.dropout_ = nn.Dropout(0.4)
        self.smax_fc = nn.Linear(400 * len(self.modals), n_classes)

    def forward(self, text_feature=None, audio_feature=None, visual_feature=None,
                speaker_tensor=None, text_length=None, **kwargs):
        features_a = []
        if 'a' in self.modals:
            audio_feature = self.linear_a(audio_feature)
            emotions_a = audio_feature
            features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a,
                                                                                                     text_length)

        features_v = []
        if 'v' in self.modals:
            visual_feature = self.linear_v(visual_feature)
            emotions_v = visual_feature
            features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v,
                                                                                                     text_length)
        features_l = []
        if 't' in self.modals:
            text_feature = self.linear_l(text_feature)
            emotions_l, hidden_l = self.lstm_l(text_feature)
            features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l,
                                                                                                     text_length)

        emotions_feat = self.graph_model(features_a, features_v, features_l, text_length, speaker_tensor)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        # log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        return self.smax_fc(emotions_feat), None
        # return log_prob


class MMGCNTrainer(MMBaseTrainer):

    def imodels(self, params: ParamsType):
        super().imodels(params)
        self.model = MMGCNModule(hidden_text=params.hidden_text,
                                 hidden_visual=params.hidden_visual,
                                 hidden_audio=params.hidden_audio,
                                 n_speakers=params.n_speakers,
                                 n_classes=params.n_classes, modals=params.modality)
        self.optim = params.optim.build(self.model.parameters())
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


main = partial(main, MMGCNTrainer, ParamsType)
