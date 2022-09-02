import torch
import numpy as np


def simple_batch_graphify(features, lengths):
    """
    Create flattened feature with `lengths`(attention) mask

    :param features: [text_length, batch_size, feature_dim]
    :param lengths: [batch_size, ]
    :return: [sum(text_length), feature_dim]
    """
    node_features = []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    return node_features, None, None, None, None


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()

            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.to('cpu')
        edge_index = edge_index.to('cpu')
        edge_norm = edge_norm.to('cpu')
        edge_type = edge_type.to('cpu')

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(l, window_past, window_future):
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
