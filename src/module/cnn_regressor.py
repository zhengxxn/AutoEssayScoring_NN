import torch
import torch.nn as nn
import torch.nn.functional as f

from allennlp.modules.highway import Highway


class CNNRegressor(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self,
                 input_dim,
                 kernel_nums,
                 kernel_sizes: list,
                 max_kernel_size=50,
                 dropout_rate=0.5):

        super().__init__()

        self.convs = nn.ModuleList(
                        [nn.Conv2d(in_channels=1,
                                   out_channels=num,
                                   kernel_size=(width, input_dim)) for (num, width) in zip(kernel_nums, kernel_sizes)],
                        )

        # self.bias = [nn.Parameter(torch.zeros())]
        self.highway_layer = Highway(input_dim=sum(kernel_nums), num_layers=1)

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.feedforward_layer = nn.Linear(sum(kernel_nums), 1)

        self.max_kernel_size = max_kernel_size

    def forward(self, x):
        # x : [batch size, seq len, input dim]
        if x.size(1) < self.max_kernel_size:
            pd = [0, 0, 0, self.max_kernel_size - x.size(1)]

        # [batch size, max seq len, input dim]
            x = f.pad(x, pd, 'constant', 0)

        # x : [batch size, kernel num, max seq len, input dim]
        x = x.unsqueeze(1)

        # x : [batch size, kernel num, max seq_len - width]
        x = [torch.relu(conv(x).squeeze(-1)) for conv in self.convs]

        x = [torch.max_pool1d(x_, x_.size(-1)).squeeze(-1) for x_ in x]

        # [batch size, sum(kernel_num)]
        x = torch.cat(x, dim=-1)

        x = self.highway_layer.forward(x)

        # [batch size, 1]
        # logit = torch.sigmoid(self.feedforward_layer(self.dropout_layer(x)))
        logit = self.feedforward_layer(self.dropout_layer(x))
        # [batch size]
        return logit
