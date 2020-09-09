import torch
import torch.nn as nn
from transformers import *
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding
from module.global_attention import GlobalAttention
from allennlp.nn.util import get_mask_from_sequence_lengths
from module.rnn_encoder import RNNEncoder


class BertGlobalAttentionRegressor(nn.Module):

    def __init__(self, bert_pretrained_weights):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_weights)
        self.positional_encoding = PositionalEncoding(input_dim=768)

        self.linear_layer = nn.Linear(768 + 5 + 300, 1, bias=False)
        self.dropout_layer = nn.Dropout(0.6)

        self.criterion = nn.MSELoss(reduction='sum')

        self.manual_feature_layer = nn.Linear(27, 5)

        self.doc_global_attention = GlobalAttention(hid_dim=768, key_size=768)
        self.segment_encoder = RNNEncoder(embedding_dim=768, hid_dim=150, num_layers=1, dropout_rate=0.5)

        nn.init.uniform_(self.linear_layer.weight.data, -0.1, 0.1)

    def forward(self, inputs, mask, sent_counts, sent_lens,
                prompt_inputs,
                prompt_mask,
                prompt_sent_counts,
                prompt_sent_lens,
                min_score, max_score, manual_feature,
                label=None, domain_label=None, alpha=0.05):
        """

        :param alpha:
        :param domain_label:
        :param manual_feature: [batch size]
        :param max_score: [batch size]
        :param min_score: [batch size]
        :param prompt_sent_lens: [batch size, max sent count]
        :param prompt_sent_counts: [batch size]
        :param prompt_inputs:   [batch size, max sent count, max sent len]
        :param prompt_mask: [batch size, max sent count, max sent len]
        :param inputs:  [batch size, max sent count, max sent len]
        :param mask:    [batch size, max sent count, max sent len]
        :param sent_counts: [batch size]
        :param sent_lens: [batch size, max sent count]
        :param label: [batch size]
        :return:
        """
        batch_size = inputs.shape[0]
        max_sent_count = inputs.shape[1]
        max_sent_length = inputs.shape[2]

        inputs = inputs.view(-1, inputs.shape[-1])
        mask = mask.view(-1, mask.shape[-1])

        # [batch size * max sent len, hid size]
        last_hidden_states = self.bert(input_ids=inputs, attention_mask=mask)[0]
        last_hidden_states = last_hidden_states.view(batch_size, max_sent_count, max_sent_length, -1)
        last_hidden_states = self.dropout_layer(last_hidden_states)

        docs = []
        lens = []
        doc_segments = []
        for i in range(0, batch_size):
            doc = []
            doc_segment = []
            sent_count = sent_counts[i]
            sent_len = sent_lens[i]

            for j in range(sent_count):
                length = sent_len[j]
                cur_sent = last_hidden_states[i, j, :length, :]
                mean_cur_sent = torch.mean(cur_sent, dim=0)
                # print('cur sent shape', cur_sent.shape)
                doc.append(cur_sent)
                doc_segment.append(mean_cur_sent.unsqueeze(0))

            # [1, len, hid size]
            doc_vec = torch.cat(doc, dim=0).unsqueeze(0)
            doc_vec = self.positional_encoding.forward(doc_vec)

            lens.append(doc_vec.shape[1])
            # print(i, 'doc shape', doc_vec.shape)
            docs.append(doc_vec)
            doc_segments.append(doc_segment)

        batch_max_len = max(lens)
        for i, doc in enumerate(docs):
            if doc.shape[1] < batch_max_len:
                pd = (0, 0, 0, batch_max_len - doc.shape[1])
                m = nn.ConstantPad2d(pd, 0)
                doc = m(doc)

            docs[i] = doc

        # [batch size, bert embedding dim]
        docs = torch.cat(docs, 0)
        docs_mask = get_mask_from_sequence_lengths(torch.tensor(lens), max_length=batch_max_len).to(docs.device)
        # [batch size, max seq len]

        doc_weights = self.doc_global_attention(docs, docs_mask)
        doc_vec = torch.bmm(doc_weights.unsqueeze(1), docs).squeeze(1)
        doc_feature = self.dropout_layer(torch.tanh(doc_vec))
        manual_feature = torch.tanh(self.manual_feature_layer(self.dropout_layer(manual_feature)))

        # rnn segments encoder
        sorted_index = sorted(range(len(sent_counts)), key=lambda i: sent_counts[i], reverse=True)
        max_count = max_sent_count
        for idx, doc in enumerate(doc_segments):
            for i in range(max_count-len(doc)):
                doc.append(torch.zeros_like(doc[0]))
            doc_segments[idx] = torch.cat(doc, dim=0).unsqueeze(0)
        doc_segments = torch.cat(doc_segments, dim=0)

        sorted_doc_segments = doc_segments[sorted_index]
        sorted_batch_counts = sent_counts[sorted_index]
        final_hidden_states = self.segment_encoder(sorted_doc_segments, sorted_batch_counts)['final_hidden_states']
        final_hidden_states[sorted_index] = final_hidden_states
        final_hidden_states = torch.tanh(final_hidden_states)
        final_hidden_states = self.dropout_layer(final_hidden_states)

        feature = torch.cat([doc_feature, manual_feature, final_hidden_states], dim=-1)

        grade = self.linear_layer(feature)

        if label is not None:

            label = (label.type_as(grade) - min_score.type_as(grade)) / (max_score.type_as(grade) - min_score.type_as(grade))
            loss = self.criterion(input=grade.contiguous().view(-1),
                                  target=label.type_as(grade).contiguous().view(-1))
        else:
            loss = None

        prediction = grade * (max_score.type_as(grade) - min_score.type_as(grade)) + min_score.type_as(grade)
        prediction = prediction[:, 0]
        return {'loss': loss, 'prediction': prediction}