import torch
import torch.nn as nn


class GlobalAttention(nn.Module):
    """
    Train a Global Variable - energy_layer, size : [hid dim, 1]
    to Attend the keys (hidden states of encoder)
    """

    def __init__(self,
                 hid_dim,
                 key_size):
        super(GlobalAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hid_dim, bias=False)
        self.energy_layer = nn.Linear(hid_dim, 1, bias=False)

        nn.init.uniform_(self.key_layer.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.energy_layer.weight.data, -0.1, 0.1)

    def forward(self, key, mask):
        """

        :param key: [batch size, len, key size]
        :param mask: [batch size, len]
        :return: attention weights that the trained global variable attend to the key
        """

        # [batch size, len, key size] -> [*, *, hid dim]
        key_energy = self.key_layer(key)

        # attention: [*, *, hid dim] -> [*, *, 1]
        scores = self.energy_layer(torch.tanh(key_energy))
        # [*, *, 1] -> [*, *]
        scores = scores.squeeze(2)

        # mask
        scores.data.masked_fill_(mask == 0, -float('inf'))
        weights = torch.softmax(scores, dim=-1)

        # [batch size, seq len]
        return weights





