import torch
import numpy as np


def transformer_batch_graphify(features: torch.Tensor,
                               attention_mask: torch.Tensor,
                               speaker_tensor: torch.Tensor,
                               wp=5, wf=5,
                               **kwargs):
    """

    :param features: [bs, maxlength, feature fim]
    :param attention_mask: [bs, maxlength]
    :param speaker_tensor: [bs, maxlength]
    :param wp:
    :param wf:
    :param edge_type_to_idx:
    :return:
    feature: [sum(attention_mask), wp+wf, feature dim]
    type_ids: [sum(attention_mask), wp+wf]
    attention_mask: [sum(attention_mask), wp+wf]
    """
    attention_mask = attention_mask.bool()
    flatten_feature = features[attention_mask]  # [sum(attention_mask), feature dim]
    flatten_attention_mask = attention_mask[attention_mask]  # [sum(attention_mask),]
    flatten_speaker_tensor = speaker_tensor[attention_mask]  # [sum(attention_mask),]

    device = features.device
    bs, fdim = flatten_feature.shape

    pb = torch.zeros(wp, fdim, device=device)
    pa = torch.zeros(wf, fdim, device=device)
    flatten_feature = torch.cat([pb, flatten_feature, pa])

    pb = torch.zeros(wp, device=device)
    pa = torch.zeros(wf, device=device)
    flatten_attention_mask = torch.cat([pb, flatten_attention_mask, pa])
    flatten_speaker_tensor = torch.cat([pb, flatten_speaker_tensor, pa])

    index = torch.arange(bs)[:, None].repeat(1, wp + wf)
    idx_offset = torch.arange(wp + wf)[None, :]
    index = index + idx_offset

    graph_feature = flatten_feature[index]  # [sum(attention_mask), wp+wf, fdim]
    graph_attention_mask = flatten_attention_mask[index]
    graph_speaker_tensor = flatten_speaker_tensor[index]

    return graph_feature, graph_attention_mask, graph_speaker_tensor


def transformer_batch_graphify(features: torch.Tensor,
                               attention_mask: torch.Tensor,
                               speaker_tensor: torch.Tensor,
                               wp=5, wf=5,
                               **kwargs):
    """

    :param features: [bs, maxlength, feature fim]
    :param attention_mask: [bs, maxlength]
    :param speaker_tensor: [bs, maxlength]
    :param wp:
    :param wf:
    :param edge_type_to_idx:
    :return:
    feature: [sum(attention_mask), wp+wf, feature dim]
    type_ids: [sum(attention_mask), wp+wf]
    attention_mask: [sum(attention_mask), wp+wf]
    """
    text_length = attention_mask.sum(dim=-1).long()
    attention_mask = attention_mask.bool()
    flatten_feature = features[attention_mask]  # [sum(attention_mask), feature dim]
    flatten_attention_mask = attention_mask[attention_mask]  # [sum(attention_mask),]
    flatten_speaker_tensor = speaker_tensor[attention_mask]  # [sum(attention_mask),]

    device = features.device
    bs, fdim = flatten_feature.shape

    pb = torch.zeros(wp, fdim, device=device)
    pa = torch.zeros(wf, fdim, device=device)
    flatten_feature = torch.cat([pb, flatten_feature, pa])

    pb = torch.zeros(wp, device=device)
    pa = torch.zeros(wf, device=device)
    flatten_attention_mask = torch.cat([pb, flatten_attention_mask, pa])
    flatten_speaker_tensor = torch.cat([pb, flatten_speaker_tensor, pa])

    with torch.no_grad():
        index = torch.arange(bs)[:, None].repeat(1, wp + 1 + wf)
        idx_offset = torch.arange(wp + 1 + wf).flip(0)[None, :]
        index = index + idx_offset

        offset = 0
        for i in text_length.tolist():
            bs_index = index[offset:offset + i]
            bs_index = bs_index - torch.triu(bs_index, wp + 1)
            bs_index = bs_index - torch.tril(bs_index, wf - i)
            index[offset:offset + i] = bs_index
            offset += i

        index = index.flip(1)

    graph_feature = flatten_feature[index]  # [sum(attention_mask), wp+wf, fdim]
    graph_attention_mask = flatten_attention_mask[index]
    graph_speaker_tensor = flatten_speaker_tensor[index]

    return graph_feature, graph_attention_mask, graph_speaker_tensor


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx):
    device = features.device

    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    # edge_ind = []
    edge_index_lengths = []

    # for j in range(batch_size):
    # edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = "0"
            else:
                c = "1"
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_type, edge_index_lengths


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
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[
                        max(0, j - window_past): min(length, j + window_future + 1)
                        ]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
