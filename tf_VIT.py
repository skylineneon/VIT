'''
VIT使用的是编码器,注意力中不能加因果矩阵
'''

import torch
from torch import nn


def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    print(xq_.shape,freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]


class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 n_q_heads,
                 n_kv_heads,
                 ):
        super().__init__()

        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads

        self._group = n_q_heads // n_kv_heads

        self._head_size = input_dim // self._n_q_heads

        self._qw = nn.Linear(input_dim, self._head_size*self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._ow = nn.Linear(input_dim, input_dim)

    def forward(self, x, freq_cis):
        _bn, _seq, _ = x.shape
        _dk = self._head_size**0.5

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[:_seq])
        _k = apply_rotary_emb(_k, freq_cis[:_seq])

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        # _causul = torch.ones(_seq, _seq)
        # _causul = torch.triu(_causul, diagonal=1)
        # _causul[_causul == 1] = -torch.inf
        # _causul = _causul.to(x.device)

        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(_bn, -1, _seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(_bn, -1, _seq, self._head_size)

        _score = _q@_k.permute(0, 1, 3, 2)/_dk
        # _score = torch.softmax(_score + _causul, dim=-1)
        _score = torch.softmax(_score, dim=-1)

        _o = _score@_v

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        return self._ow(_o)


class FFN(nn.Module):

    def __init__(self, input_dim, hide_dim):
        super().__init__()

        self._w0 = nn.Linear(input_dim, hide_dim)
        self._w1 = nn.Linear(input_dim, hide_dim)
        self._w2 = nn.Linear(hide_dim, input_dim)

        self._gate = nn.SiLU()

    def forward(self, x):
        return self._w2(self._w0(x)*self._gate(self._w1(x)))


class RMSNormal(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        return self._w*x/((x**2).sum()**0.5+1e-6)


class TransformerLayer(nn.Module):
    """
    单层的Transformer结构
    """

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,):
        super().__init__()

        self._att_norm = RMSNormal(input_dim)
        self._att_layer = Attention(input_dim, n_q_heads, n_kv_heads)
        self._ffn_norm = RMSNormal(input_dim)
        self._ffn_layer = FFN(input_dim, hide_dim)

    def forward(self, x, freq_cis):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x, freq_cis)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y


class TransformerEncoder(nn.Module):
    """
        解码器
    """

    def __init__(self,
                 num_layers,  # 解码器的层数
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads) for _ in range(num_layers)]
        )

        _freq_cis = precompute_freqs_cis(input_dim//n_q_heads, max_len)

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x, self.freq_cis)
        return _x


if __name__ == '__main__':
    bn = 5000
    seq = 5
    vec = 64
    n_q_heads = 2
    n_kv_heads = 1
    n_layers = 2
    max_len = 4

    # freq_cis = precompute_freqs_cis(vec//n_q_heads, max_len)

    x = torch.randn(bn, seq, vec)

    # att = Attention(
    #     input_dim = vec,
    #     n_q_heads = n_q_heads,
    #     n_kv_heads = n_kv_heads,
    # )

    # y = att(x,freq_cis)
    # print(y.shape)

    # ffn = FFN(
    #     input_dim=vec,
    #     hide_dim=vec//2
    # )

    # y = ffn(x)
    # print(y.shape)

    # norm = RMSNormal(input_dim=vec)
    # y = norm(x)
    # print(y.shape)

    # layer = TransformerLayer(
    #     input_dim=vec,
    #     hide_dim=vec//2,
    #     n_q_heads=n_q_heads,
    #     n_kv_heads=n_kv_heads
    # )

    # y = layer(x,freq_cis)
    # print(y.shape)

    decoder = TransformerEncoder(
        num_layers=n_layers,
        input_dim=vec,
        hide_dim=vec//2,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        max_len=max_len
    )

    y = decoder(x)
    print(y.shape)
