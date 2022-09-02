# Multi-modal Emotion recognition for conversation

- COGMEN
- MMGCN
- DAG-ERC
- DialogueGCN

# COGMEN

所有模态的特征在一开始就被 concat 到一起（early fusion），过一遍 Transformer Encoder，在之后输入可以视为一个模态特征。

随后，构建一个邻接表（因为接口需要，本质还是一个图），对每一个对话，将当前对话前后的 N 句话之间建立边。

同时，边的类型为边相连结点对应的说话人 pair 的类型（speaker）。如，两个人就是一共四种类型（00,01,10,11）。

最后得到的有向属性图，和 node_feature 一起，依次通过 RGCNConv 和TransformerConv

# MMGCN

MMGCN 对每一个对话按两种方式构建图链接：

- 模态间，计算每一个 uttr 在彼此模态之间的相似度，忽略不同 uttr 彼此模态之间的相似度，得到一个 dia_len * 1 的相似度向量
- 模态内，计算每一个 uttr 之间的相似度，得到一个 dia_len * dia_len 的相似度矩阵

最终，构建成一个长宽 (dia_len * modal_len) 的 big 邻接图矩阵，可以看成是 modal_len * modal_len 的方块，每个方块是 dia_len * dia_len 的相似度矩阵。
其中，对角的相似度矩阵（modal_len 个）表示模态内的，其余的（modal_len * modal_len - modal_len 个）为模态间的。

模态内（对角）的相似度矩阵可以看成是全填满的，而模态间的矩阵只有对角线填满。

最终这样一个大图和不同模态的特征组成的大 feature 一块过 GraphConvolution

该方法没有考虑对话上下文的远近，统统按相似度的方式求解。

也就是每一个模态的每一个时间步特征，都要和同模态的上下文，不同模态同时间步的特征做 attention 操作

其实就是一个 a t v 在时间步上 concat 后的一个 Transformer 操作，attention mask 会是一个 [bs, ts, ts] 维的，因为不同时间步的不统一了

# DAG-ERC

# DialogueGCN