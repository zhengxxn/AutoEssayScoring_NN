import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self,
                 embedding_dim,
                 hid_dim,
                 num_layers=1,
                 dropout_rate=0.5):

        super(RNNEncoder, self).__init__()

        self.emb_dim = embedding_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(
                        input_size=embedding_dim,
                        hidden_size=hid_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=dropout_rate
                    )

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_embedding, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        input_embedding = self.dropout_layer(input_embedding)

        packed_input = pack_padded_sequence(input_embedding, lengths, batch_first=True)
        packed_output, h_n = self.rnn(packed_input)
        # unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # unpacked_output [batch size, seq len, num directions *  hidden size]
        # h_n [batch size, num layers * num directions, hidden size]

        fwd_final = h_n[0:h_n.size(0):2]
        bwd_final = h_n[1:h_n.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2 * hid dim]

        # context_mean = torch.sum(unpacked_output, dim=1) / lengths.unsqueeze(1).type_as(unpacked_output)
        encoder_state = {'final_hidden_states': final.squeeze(0)}
        return encoder_state
