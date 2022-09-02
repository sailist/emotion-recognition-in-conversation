import torch
import math
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple


class MLP(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,  # hidden_size
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, feature):
        return self.module(feature)


class VIB(nn.Module):
    """
    https://github.com/bojone/vib/blob/master/cnn_imdb_vib.py
    """

    def __init__(self, feature_dim, lamb=0.1):
        super().__init__()
        self.lamb = lamb
        self.to_mean = nn.Linear(feature_dim, feature_dim)
        self.to_var = nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, reduction='sum'):
        z_mean, z_log_var = self.to_mean(feature), self.to_var(feature)

        if self.training:
            kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
            if reduction == 'sum':
                kl_loss = -0.5 * kl_loss.mean(0).sum()
            u = torch.rand_like(z_mean)
        else:
            kl_loss = 0
            u = 0
        feature = z_mean + torch.exp(z_log_var / 2) * u
        return (feature, kl_loss)


class MLP2(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 mid_dim=128,
                 output_dim=64, with_bn=False, with_leakyrelu=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.BatchNorm1d(mid_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if with_leakyrelu else nn.ReLU(inplace=True),
            nn.Linear(mid_dim, output_dim),
            nn.BatchNorm1d(output_dim) if with_bn else nn.Identity(),
        )

    def forward(self, feature):
        return self.module(feature)


class NormMLP(MLP):
    def forward(self, feature):
        return F.normalize(super().forward(feature), p=2, dim=-1)


class ResidualLinear(nn.Module):
    def __init__(self, in_feature, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feature, in_feature, bias=bias)

    def forward(self, feature):
        out = self.linear(feature)
        out = out + feature
        return out


from transformers.models.bert.modeling_bert import BertAttention, BertModel, BertConfig


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 position_embedding_type=None,
                 dropout=0.1,  # default value of attention_probs_dropout_prob
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 ):
        super().__init__()
        if q_dim is None:
            q_dim = hidden_size
        if k_dim is None:
            k_dim = hidden_size

        if v_dim is None:
            v_dim = hidden_size

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q_proj = nn.Linear(q_dim, self.all_head_size)
        self.k_proj = nn.Linear(k_dim, self.all_head_size)
        self.v_proj = nn.Linear(v_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.position_embedding_type = position_embedding_type

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            *args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param query:
        :param key:
        :param value:
        :param attention_mask: should be extened attention_mask
        :param head_mask:
        :param encoder_hidden_states:
        :param past_key_value:
        :param output_attentions:
        :return:
        """
        mixed_query_layer = self.q_proj(query)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        key_layer = self.transpose_for_scores(self.k_proj(key))
        value_layer = self.transpose_for_scores(self.v_proj(value))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs)

        return outputs


@torch.no_grad()
def extends_attention_mask(attention_mask):
    assert attention_mask.dim() == 2
    extended_attention_mask = attention_mask[:, None, None, :]
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
