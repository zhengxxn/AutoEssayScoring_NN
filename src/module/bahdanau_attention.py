import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self,
                 hid_dim,
                 query_size,
                 key_size):
        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hid_dim, bias=False)
        self.query_layer = nn.Linear(query_size, hid_dim, bias=False)
        self.energy_layer = nn.Linear(hid_dim, 1, bias=False)

        nn.init.uniform_(self.key_layer.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.query_layer.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.energy_layer.weight.data, -0.1, 0.1)

    def forward(self, query, key, mask, key_energy_cache=None):

        query = query.unsqueeze(1)

        query_energy = self.query_layer(query)

        if key_energy_cache is not None:
            key_energy = key_energy_cache
        else:
            key_energy = self.key_layer(key)

        # [batch size, seq len, 1] -> [batch size, 1, seq len]
        scores = self.energy_layer(torch.tanh(query_energy + key_energy))
        scores = scores.squeeze(2)

        scores.data.masked_fill_(mask == 0, -float('inf'))
        weights = torch.softmax(scores, dim=-1)

        # [batch size, seq len]
        return weights





