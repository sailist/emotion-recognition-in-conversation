import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GraphConv
from models.rgcn import RGCNConv


class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, dropout=0.4, rnn_type='lstm'):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=dropout,
                               bidirectional=True, num_layers=2, batch_first=True)
        else:
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=dropout,
                              bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, text_len_tensor, text_tensor):
        packed = pack_padded_sequence(
            text_tensor,
            text_len_tensor.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        rnn_out, (_, _) = self.rnn(packed, None)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out


class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, n_speakers):
        super(GCN, self).__init__()
        self.num_relations = 2 * n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv2(x, edge_index)

        return x


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model):
    device = features.device

    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


class EdgeAtt(nn.Module):

    def __init__(self, g_dim, wp, wf):
        super(EdgeAtt, self).__init__()
        self.wp = wp
        self.wf = wf

        self.weight = nn.Parameter(torch.zeros((g_dim, g_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, text_len_tensor, edge_ind):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        att_matrix = torch.matmul(weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = text_len_tensor[i].item()
            alpha = torch.zeros((mx_len, 110)).to(node_features.device)
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat)
                probs = F.softmax(score, dim=-1)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)
        # alphas = torch.ones(batch_size, mx_len, 110).to(self.device)

        return alphas


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, dropout, loss_weights=True):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_size, tag_size)

    def forward(self, h, text_len_tensor):
        hidden = self.drop(F.relu(self.lin1(h)))
        logits = self.lin2(hidden)

        # log_prob = self.get_prob(h, text_len_tensor)
        # y_hat = torch.argmax(log_prob, dim=-1)

        return logits


class MaskedEmotionAtt(nn.Module):

    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s: s + cur_len, :] = out
            s += cur_len

        return ret
