"""
Reimplementation of MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation
"""
import torch
from torch.optim import lr_scheduler

from contrib.make_optim import make_optim
from lumo import Meter, callbacks, DataModule, MetricType, TrainStage
from lumo.contrib import EMA
from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
from lumo import CollateBase
from contrib.database import TableRow
from functools import partial

from .mmgcn_models import *
from .mmgcn_utils import *
from .mmbase import MMBaseTrainer, MMBaseParams, main


class MMGCNParams(MMBaseParams):

    def __init__(self):
        super().__init__()
        self.seed = 1
        self.module = None
        self.method = None

        self.epoch = 60
        self.train.batch_size = 16
        self.test.batch_size = 16

        self.num_heads = 10
        self.gnn_heads = 1
        self.confuse_matrix = True
        self.dataset = 'iemocap-cogmen-6'
        self.optim = self.OPTIM.create_optim('Adam', lr=0.0003, weight_decay=3e-5)
        self.split_params = False
        self.train.num_workers = 2
        self.test.num_workers = 2
        self.ema = True

        self.sche_type = self.choice('cos', 'gamma')
        self.warmup_epochs = 0
        self.pretrain_path = None

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
                speaker_feature=None, text_length=None, **kwargs):
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

        emotions_feat = self.graph_model(features_a, features_v, features_l, text_length, speaker_feature)
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


class MMGCNCollate(CollateBase):
    def __init__(self, params: ParamsType):
        super().__init__(params)
        self.modalities = params.modality
        self.n_speakers = params.n_speakers

    def __call__(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s['text']) for s, in samples]).long()
        mx = torch.max(text_len_tensor).item()

        text_feature = []
        audio_feature = []
        visual_feature = []
        speaker_tensor = torch.zeros((batch_size, mx, self.n_speakers)).long()
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
            tmp_a = []
            tmp_v = []
            tmp_t = []

            for t, a, v in zip(textual, audio, visual):
                tmp_a.append(torch.from_numpy(a))
                tmp_v.append(torch.from_numpy(v))
                tmp_t.append(torch.from_numpy(t))

            easy = [audio_feature, visual_feature, text_feature]
            """
            Reimplementation of paper COGMEN COntextualized GNN based Multimodal Emotion recognitioN

            接下来的两条路线
             - [x] 拆分特征，和合并特征双路前后融合同时 v1x
             - speaker 区分构建两路 transformer（一路 context，一路同说话人）
             - 自监督损失函数

             - 不同的结构会导致不同的微妙差异
             - 也就是不同的结构对不同类别识别的倾向性不同
             - 因此可以设计一个树状结构？做一个双路特征加权？
            """
            from functools import partial

            import torch
            from torch import nn
            from torch.nn import functional as F
            from torch.optim import lr_scheduler

            from contrib.make_optim import make_optim
            from contrib.nn import TransformerEncoderLayer
            from lumo import CollateBase
            from lumo import Meter, callbacks, DataModule, MetricType, TrainStage
            from lumo.contrib import EMA
            from mmdatasets.erc_dataset import get_train_dataset, get_test_dataset, get_val_dataset
            from track_mm.cogmen_utils import batch_graphify, extends_attention_mask, transformer_batch_graphify
            from track_mm.mmbase import MMBaseTrainer, MMBaseParams, main
            from track_mm.cogmen import COGMENDM
            from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertConfig
            from transformers.modeling_utils import ModelOutput
            from models.components import MLP, MLP2, ResidualLinear
            from models.losses import focal_loss

            class COGMENParams(MMBaseParams):

                def __init__(self):
                    super().__init__()
                    self.seed = 1

                    self.train.batch_size = 32
                    self.val.batch_size = 32
                    self.test.batch_size = 32

                    self.global_layers = 2
                    self.local_layers = 6

                    self.num_heads = 17

                    self.wp = 5
                    self.wf = 5

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
                    super(COGMENParams, self).iparams()

            ParamsType = COGMENParams

            class COGMENModule(nn.Module):
                def __init__(self,
                             input_size,
                             hidden_size,

                             num_head,
                             n_speakers,
                             n_classes,
                             context=None,
                             dropout=0.5,
                             global_layers=2,
                             local_layers=6,
                             text_dim=100,
                             audio_dim=100,
                             visual_dim=100,
                             ):
                    super().__init__()
                    if context is None:
                        context = [5, 5]

                    # past, future
                    self.wp, self.wf = context

                    # num_head = 16
                    # mdims = [audio_dim, text_dim, visual_dim]
                    # self.linear_lis = nn.ModuleList([
                    #     nn.Linear(mdim, 512) for mdim in mdims
                    # ])
                    # input_size = 512 * 3
                    # num_head = 24

                    find_head = False
                    for h in range(6, num_head):
                        if input_size % h == 0:
                            num_head = h
                            find_head = True
                            break
                    assert find_head

                    config = BertConfig(hidden_size=input_size, num_hidden_layers=global_layers,
                                        num_attention_heads=num_head)
                    self.encoder = BertEncoder(config)

                    self.linaer = nn.Linear(
                        input_size, hidden_size, bias=True
                    )
                    self.linear_2 = nn.Linear(input_size, hidden_size, bias=True)

                    gcn_config = BertConfig(hidden_size=hidden_size, num_hidden_layers=local_layers,
                                            num_attention_heads=16)

                    self.gcn = BertEncoder(gcn_config)

                    self.cls = nn.Sequential(*[
                        # MaskedEmotionAtt(100),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(p=dropout),
                        ResidualLinear(hidden_size),
                        nn.Linear(hidden_size, n_classes),
                    ])

                    # self.cls = MLP2(hidden_size, hidden_size, n_classes, with_bn=True)

                    self.cls_id = n_speakers
                    self.speaker_embedding = nn.Embedding(n_speakers + 1, hidden_size)
                    # self.cls_embedding = nn.Embedding(2, hidden_size)

                    for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)

                def forward(self,
                            input_tensor,
                            attention_mask,
                            speaker_tensor,
                            text_feature,
                            audio_feature,
                            visual_feature,
                            *args,
                            **kwargs
                            ):
                    """

                    :param input_tensor: [bs, mx, feature_dim]
                    :param attention_mask: [bs, mx]
                    :param speaker_tensor: [bs, mx]
                    :param text_feature: [bs, mx, feature_dim]
                    :param audio_feature: [bs, mx, feature_dim]
                    :param visual_feature: [bs, mx, feature_dim]
                    :param args:
                    :param kwargs:
                    :return:
                    """
                    device = input_tensor.device

                    # audio_feature = self.linear_a(audio_feature)
                    # features = [self.linear_lis[i](feat) for i, feat in enumerate([
                    #     audio_feature,
                    #     text_feature,
                    #     visual_feature,
                    # ])]
                    # text_feature, audio_feature, visual_feature = features
                    node_feature = torch.cat([audio_feature, text_feature, visual_feature], dim=-1)
                    # node_feature = input_tensor

                    node_feature = self.encoder.forward(node_feature,
                                                        attention_mask=extends_attention_mask(
                                                            attention_mask)).last_hidden_state
                    node_feature = self.linaer(node_feature)  # + self.linaer_2(input_tensor)
                    # node_feature =

                    res = transformer_batch_graphify(node_feature,
                                                     attention_mask=attention_mask,
                                                     speaker_tensor=speaker_tensor, wp=self.wp, wf=self.wf)

                    graph_feature, graph_attention_mask, graph_speaker_tensor = res

                    graph_out = self.gcn.forward(graph_feature,
                                                 attention_mask=extends_attention_mask(graph_attention_mask))
                    features = graph_out.last_hidden_state[:, self.wp]
                    features = torch.cat([
                        features,
                        # self.linear_2(input_tensor)[attention_mask.bool()],
                        # node_feature[attention_mask.bool()],
                    ], dim=-1)

                    logits = self.cls(features)
                    return COGMENModelOutput(logits=logits,
                                             features=features,
                                             graph_attention_mask=graph_attention_mask,
                                             graph_speaker_tensor=graph_speaker_tensor)

            class COGMENModelOutput(ModelOutput):
                logits: torch.Tensor
                spk_logits: torch.Tensor
                features: torch.Tensor
                graph_attention_mask: torch.Tensor
                graph_speaker_tensor: torch.Tensor

            class COGMENTrainer(MMBaseTrainer):

                def imodels(self, params: ParamsType):
                    super().imodels(params)
                    self.model = COGMENModule(input_size=params.hidden_all,
                                              hidden_size=128,
                                              num_head=params.num_heads,
                                              n_speakers=params.n_speakers,
                                              n_classes=params.n_classes,
                                              text_dim=params.hidden_text,
                                              audio_dim=params.hidden_audio,
                                              visual_dim=params.hidden_visual,
                                              )
                    self.optim = make_optim(self.model, params.optim, split=params.split_params)
                    self.logger.raw(self.optim)
                    self.to_device()

                    self.lr_sche = lr_scheduler.ReduceLROnPlateau(self.optim, "min")

                    if params.ema:
                        self.ema_model = EMA(self.model, alpha=0.999)

                def to_logits(self, xs):
                    return self.model(**xs).logits

                def to_ema_logits(self, xs):
                    return self.ema_model(**xs).logits

                def train_step(self, batch, params: ParamsType = None) -> MetricType:
                    super().train_step(batch, params)
                    meter = Meter()

                    ys = batch['label']
                    outputs = self.model.forward(**batch)
                    graph_out, features = outputs.logits, outputs.features

                    # graph_attention_mask = outputs.graph_attention_mask
                    # spk_logits = outputs.spk_logits[graph_attention_mask.bool()]
                    # graph_speaker_tensor = outputs.graph_speaker_tensor[graph_attention_mask.bool()]
                    # Lspk = F.cross_entropy(spk_logits, graph_speaker_tensor.long())

                    Lcs = F.cross_entropy(graph_out, ys)
                    Lall = Lcs

                    self.optim.zero_grad()
                    self.accelerate.backward(Lall)
                    self.optim.step()

                    if params.ema:
                        self.ema_model.step()

                    with torch.no_grad():
                        meter.Lall = Lall
                        meter.Lcs = Lcs
                        meter.Acc = torch.eq(graph_out.argmax(dim=-1), ys).float().mean()

                    return meter

            main = partial(main, COGMENTrainer, ParamsType, COGMENDM)
            for ii, tmp in enumerate([tmp_a, tmp_v, tmp_t]):
                tmp = torch.stack(tmp)
                tmp = torch.cat([tmp, torch.zeros(mx - len(tmp), tmp.shape[1], dtype=tmp.dtype)])
                easy[ii].append(tmp)
            speaker_tensor[i, :cur_len] = torch.tensor(speakers)

            labels.extend(label)

        label_tensor = torch.tensor(labels).long()
        data = {
            "text_length": text_len_tensor,  # K * bs * feature_dim
            "text_feature": torch.stack(text_feature, dim=1) if 't' in self.modalities else None,
            "audio_feature": torch.stack(audio_feature, dim=1) if 'a' in self.modalities else None,
            "visual_feature": torch.stack(visual_feature, dim=1) if 'v' in self.modalities else None,
            "speaker_feature": speaker_tensor.transpose(0, 1),  # K * bs * 2, onehot
            "label": label_tensor,
            "utterance_texts": utterances,
        }
        return data


class MMGCNDM(DataModule):
    def __init__(self, params: ParamsType = None):
        super().__init__(params)

    def idataloader(self, params: ParamsType = None, stage: TrainStage = None):
        collate_fn = MMGCNCollate(params)
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


main = partial(main, MMGCNTrainer, ParamsType, MMGCNDM)
