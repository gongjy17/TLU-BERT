import math
from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    """多头注意力机制
        - key_size: 键向量的大小，默认为768。
        - query_size: 查询向量的大小，默认为768。
        - value_size: 值向量的大小，默认为768。
        - num_hiddens: 隐藏层的维数。
        - num_heads: 注意力机制的头数。
        - dropout: 用于层间的dropout比例。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,`num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class DotProductAttention(nn.Module):
    """点积注意力
        - dropout: 用于层间的dropout比例。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

def masked_softmax(X, valid_lens):
    """
    通过在最后一个轴上掩盖元素来执行softmax操作。
    """
    # `X`: 3维张量, `valid_lens`: 1维或2维张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果valid_lens是1维的，则将其重复扩展以匹配X的形状
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 如果valid_lens是2维的，则将其重塑为1维
            valid_lens = valid_lens.reshape(-1)
        # 在最后一个轴上，用一个非常大的负数替换被掩盖的元素，
        # 这个大的负数的指数结果为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # 将X重塑回原来的形状，并执行softmax操作
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def sequence_mask(X, valid_len, value=0):

    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def transpose_qkv(X, num_heads):
    """
    为多个注意力头的并行计算进行转置操作。
    """
    # 输入 `X` 的形状：
    # (`batch_size`, 查询或键值对的数量, `num_hiddens`)。
    # 输出 `X` 的形状：
    # (`batch_size`, 查询或键值对的数量, `num_heads`,`num_hiddens` / `num_heads`)
    # 重新塑形X以分割隐藏维度到多个头
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出 `X` 的形状：
    # (`batch_size`, `num_heads`, 查询或键值对的数量,
    # `num_hiddens` / `num_heads`)
    # 使用permute对X进行转置以便将头维度移到前面
    X = X.permute(0, 2, 1, 3)

    # `output` 的形状：
    # (`batch_size` * `num_heads`, 查询或键值对的数量, `num_hiddens` / `num_heads`)
    # 重塑输出以合并批次和头维度，为点积注意力做准备
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """
    Reverse the operation of `transpose_qkv`.
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
