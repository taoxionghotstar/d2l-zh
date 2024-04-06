import math
import torch
from torch import nn
from d2l import torch as d2l


# X.shape = [batch_size, num_steps_1, num_steps_2]
# valid_lens.shape = [2]
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    # queries.shape = [batch_size, num_steps_q, query_size]
    # keys.shape = [batch_size, num_steps_kv, key_size]
    # values.shape = [batch_size, num_steps_kv, value_size]
    def forward(self, queries, keys, values, valid_lens):
        # queries.shape = [batch_size, num_steps_q, num_hiddens]
        # keys.shape = [batch_size, num_steps_kv, num_hiddens]
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries.unsqueeze(2).shape = [batch_size, num_steps_q, 1, num_hiddens]
        # keys.unsqueeze(1).shape = [batch_size, 1, num_steps_kv, num_hiddens]
        # features.shape = [batch_size, num_steps_q, num_steps_kv, num_hiddens]，这里使用了广播
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v(features).shape = [batch_size, num_steps_q, num_steps_kv, 1]
        # scores.shape = [batch_size, num_steps_q, num_steps_kv]
        scores = self.w_v(features).squeeze(-1)
        # self.attention_weights.shape = [batch_size, num_steps_q, num_steps_kv]
        self.attention_weights = masked_softmax(scores, valid_lens)
        # self.dropout(self.attention_weights).shape = [batch_size, num_steps_q, num_steps_kv]
        # values.shape = [batch_size, num_steps_kv, value_size]
        # 输出的shape为[batch_size, num_steps_q, value_size]
        return torch.bmm(self.dropout(self.attention_weights), values)


# queries.shape = (batch_size(2), num_steps_q(1), query_size(20))
# keys.shape = (batch_size(2), num_steps_kv(10), key_size(2))
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values.shape = (batch_size(2), num_steps_kv(10), value_size(4))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)