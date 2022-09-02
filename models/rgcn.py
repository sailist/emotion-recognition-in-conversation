"""
Copied from torch_geometric 1.4.2, which is a version that contains `edge_norm` argument in `RGCNCONV.forward`

used in DialogueGCN
"""
import inspect
import math
from collections import OrderedDict
import torch
from torch.nn import Parameter as Param

import torch_scatter


def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'min', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0

    return out


msg_special_args = set([
    'edge_index',
    'edge_index_i',
    'edge_index_j',
    'size',
    'size_i',
    'size_j',
])

aggr_special_args = set([
    'index',
    'dim_size',
])

update_special_args = set([])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """

    def __init__(self, aggr='add', flow='source_to_target', node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (f'Encountered node tensor with size '
                 f'{tensor.size(self.node_dim)} in dimension {self.node_dim}, '
                 f'but expected size {size[index]}.'))

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out['edge_index'] = edge_index
        out['edge_index_i'] = edge_index[i]
        out['edge_index_j'] = edge_index[j]
        out['size'] = size
        out['size_i'] = size[i]
        out['size_j'] = size[j]

        # Add special aggregate arguments.
        out['index'] = out['edge_index_i']
        out['dim_size'] = out['size_i']

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """

        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs, **kwargs):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
