import torch
from torch import nn
from attention import *
from d2l import torch as d2l

#@save
class BERTModel(nn.Module):
    """
    BERT模型
        - vocab_size: 词汇表大小，确定了词嵌入的维度。
        - num_hiddens: 隐藏层的维数。
        - norm_shape: 归一化层的形状。
        - ffn_num_input: 前馈网络的输入维度。
        - ffn_num_hiddens: 前馈网络隐藏层的维度。
        - num_heads: 注意力机制的头数。
        - num_layers: 编码器中的层数。
        - dropout: 用于层间的dropout比例。
        - max_len: 序列的最大长度，默认为1000。
        - key_size: 键向量的大小，默认为768。
        - query_size: 查询向量的大小，默认为768。
        - value_size: 值向量的大小，默认为768。
        - hid_in_features: 用于下一句预测的多层感知机分类器的隐藏层的维数。
        - mlm_in_features: 掩蔽语言模型的隐藏层维数。
        - nsp_in_features: 下一句预测模型的隐藏层维数。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

class BERTEncoder(nn.Module):
    """
    BERT编码器:
        - vocab_size: 词汇表大小，确定了词嵌入的维度。
        - num_hiddens: 隐藏层的维数。
        - norm_shape: 归一化层的形状。
        - ffn_num_input: 前馈网络的输入维度。
        - ffn_num_hiddens: 前馈网络隐藏层的维度。
        - num_heads: 注意力机制的头数。
        - num_layers: 编码器中的层数。
        - dropout: 用于层间的dropout比例。
        - max_len: 序列的最大长度，默认为1000。
        - key_size: 键向量的大小，默认为768。
        - query_size: 查询向量的大小，默认为768。
        - value_size: 值向量的大小，默认为768。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self,vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

        def forward(self, tokens, segments, valid_lens):
            # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
            X = self.token_embedding(tokens) + self.segment_embedding(segments)
            X = X + self.pos_embedding.data[:, :X.shape[1], :]
            for blk in self.blks:
                X = blk(X, valid_lens)
            return X

class EncoderBlock(nn.Module):
    """
    Transformer编码器:
        - key_size: 键向量的大小，默认为768。
        - query_size: 查询向量的大小，默认为768。
        - value_size: 值向量的大小，默认为768。
        - num_hiddens: 隐藏层的维数。
        - norm_shape: 归一化层的形状。
        - ffn_num_input: 前馈网络的输入维度。
        - ffn_num_hiddens: 前馈网络隐藏层的维度。
        - num_heads: 注意力机制的头数。
        - dropout: 用于层间的dropout比例。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class AddNorm(nn.Module):
    """
    带有层规范化的残差连接
        - normalized_shape: 归一化层的形状。
        - dropout: 用于层间的dropout比例。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    """
    逐位前馈网络
        - ffn_num_input: 前馈网络的输入维度。
        - ffn_num_hiddens: 前馈网络隐藏层的维度。
        - ffn_num_outputs: 前馈网络输出维度。
        - **kwargs: 其他关键字参数，主要用于nn.Module的初始化。
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)