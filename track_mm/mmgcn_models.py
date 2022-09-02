from torch import nn
import numpy as np
import math
from torch.nn import functional as F, Parameter
import torch


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            M_ = M.permute(1, 2, 0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_ * mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked / alpha_sum
        else:
            M_ = M.transpose(0, 1)
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)
            M_x_ = torch.cat([M_, x_], 2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]
        return attn_pool, alpha


class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type == 'av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == 'general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim * 3, 1)
            self.transform_al = nn.Linear(mem_dim * 3, 1)
            self.transform_vl = nn.Linear(mem_dim * 3, 1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) != 0 else a
        v = self.dropoutv(v) if len(v) != 0 else v
        l = self.dropoutl(l) if len(l) != 0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l], dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa * (self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l], dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv * (self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l, hma, hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l, hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l, hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 't' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 't' not in modals:
                    return h_av
            if 'a' in modals and 't' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a * l], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 't' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v * l], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl], dim=-1)


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn -> edge_idn是边的index的集合
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':

            scale = self.scalar(M)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            mask = (torch.ones(alpha.size()) * 1e-10).detach().to(alpha.device)
            mask_copy = (torch.zeros(alpha.size())).detach().to(alpha.device)

            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    edge_ind_.append([i, x[0], x[1]])

            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True,
                                 device=M.device)

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True,
                                 device=M.device)

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


class GCNII_lyc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue,
                 new_graph=False):
        super(GCNII_lyc, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel, adj=None):
        if adj is None:
            if self.new_graph:
                adj = self.message_passing_relation_graph(x, dia_len)
            else:
                adj = self.message_passing_wo_speaker(x, dia_len, topicLabel)
        else:
            adj = adj
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length)
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start + dia_len[i], start:start + dia_len[i]] = sub_adj
            start += dia_len[i]

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in
                                       range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in
                        list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj


class MMGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant,
                 return_feature, use_residue, new_graph='full', n_speakers=2, modals=['a', 'v', 't'], use_speaker=True,
                 use_modal=False):
        super(MMGCN, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
                                   dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                                   return_feature=return_feature, use_residue=use_residue)
        self.a_fc = nn.Linear(a_dim, n_dim)
        self.v_fc = nn.Linear(v_dim, n_dim)
        self.l_fc = nn.Linear(l_dim, n_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim * 3 + nhidden * 3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def forward(self, a, v, l, dia_len, qmask):
        """

        :param a: [sum(dia_len), feature_dim]
        :param v: [sum(dia_len), feature_dim]
        :param l: [sum(dia_len), feature_dim]
        :param dia_len: [batch_size, ]
        :param qmask: [batch_size, speaker_dim]
        :return:
        """
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)  # [sum(dia_len), speaker_dim]
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)  # [sum(dia_len), embedding_dim]
        if self.use_speaker:
            if 't' in self.modals:
                l += spk_emb_vector
        if self.use_modal:  # modal type-embedding, default is False
            emb_idx = torch.LongTensor([0, 1, 2], device=self.device)
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 't' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        adj = self.create_big_adj(a, v, l, dia_len, self.modals)

        if len(self.modals) == 3:
            features = torch.cat([a, v, l], dim=0)
        elif 'a' in self.modals and 'v' in self.modals:
            features = torch.cat([a, v], dim=0)
        elif 'a' in self.modals and 't' in self.modals:
            features = torch.cat([a, l], dim=0)
        elif 'v' in self.modals and 't' in self.modals:
            features = torch.cat([v, l], dim=0)
        else:
            return NotImplementedError
        features = self.graph_net(features, None, qmask, adj)
        all_length = l.shape[0] if len(l) != 0 else a.shape[0] if len(a) != 0 else v.shape[0]
        if len(self.modals) == 3:
            features = torch.cat(
                [features[:all_length], features[all_length:all_length * 2], features[all_length * 2:all_length * 3]],
                dim=-1)
        else:
            features = torch.cat([features[:all_length], features[all_length:all_length * 2]], dim=-1)
        if self.return_feature:
            return features
        else:
            return F.softmax(self.final_fc(features), dim=-1)

    def create_big_adj(self, a, v, l, dia_len, modals):
        modal_num = len(modals)
        all_length = l.shape[0] if len(l) != 0 else a.shape[0] if len(a) != 0 else v.shape[0]
        adj = torch.zeros((modal_num * all_length, modal_num * all_length), device=self.device)
        if len(modals) == 3:
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 't' in modals:
            features = [a, l]
        elif 'v' in modals and 't' in modals:
            features = [v, l]
        else:
            return NotImplementedError
        start = 0
        for i in range(len(dia_len)):  # each dialog
            sub_adjs = []
            for j, x in enumerate(features):  # each modal
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]  # modal feature
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)),
                                               dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix) / np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix  # dialog-wise similarity
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i].detach().cpu().numpy()))
            for m in range(modal_num):  # pair-wise modal interact
                for n in range(modal_num):
                    m_start = start + all_length * m
                    n_start = start + all_length * n
                    if m == n:
                        adj[m_start:m_start + dia_len[i], n_start:n_start + dia_len[i]] = sub_adjs[
                            m]  # put each modal adj in final big adj
                    else:  # different modal interact, only calculate similarity between same uttr
                        modal1 = features[m][start:start + dia_len[i]]  # length, dim
                        modal2 = features[n][start:start + dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(
                            torch.sum(modal1.mul(modal1), dim=1))  # dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(
                            torch.sum(modal2.mul(modal2), dim=1))  # dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1)  # length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim) / np.pi
                        idx = dia_idx.copy()
                        idx[0] += int(m_start)
                        idx[1] += int(n_start)
                        adj[idx] = dia_sim

            start += dia_len[i]

        d = adj.sum(1)  # degree
        # D denotes the diagonal degree matrix of graph G
        D = torch.diag(torch.pow(d, -0.5))
        # A(adj) denotes the adjacency matrix

        # let P ̃ be the renormalized graph Laplacian matrix
        adj = D.mm(adj).mm(D)  # P ̃

        return adj
