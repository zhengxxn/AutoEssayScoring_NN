import torch
import torch.nn as nn
from transformers import *
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding


class BertSingleClassifier(nn.Module):

    def __init__(self, bert_pretrained_weights, num_class):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_weights)
        # self.positional_encoding = PositionalEncoding(input_dim=768)

        # self.linear_doc = nn.Linear(768, 768)
        # self.linear_prompt = nn.Linear(768, 768)

        self.linear_layer = nn.Linear(768 * 2, num_class)
        self.dropout_layer = nn.Dropout(0.2)
        self.criterion = nn.NLLLoss(reduction='sum')

        nn.init.uniform_(self.linear_layer.weight.data, -0.1, 0.1)
        nn.init.zeros_(self.linear_layer.bias.data)

    def forward(self, inputs, mask, sent_counts, sent_lens,
                prompt_inputs,
                prompt_mask,
                prompt_sent_counts,
                prompt_sent_lens,
                features=None,
                label=None):
        """

        :param prompt_sent_lens:
        :param prompt_sent_counts:
        :param prompt_inputs:
        :param prompt_mask:
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

        prompt_inputs = prompt_inputs.view(-1, prompt_inputs.shape[-1])
        prompt_mask = prompt_mask.view(-1, prompt_mask.shape[-1])
        prompt_hidden_states = self.bert(input_ids=prompt_inputs, attention_mask=prompt_mask)[0]
        prompt_hidden_states = self.dropout_layer(prompt_hidden_states)

        docs = []
        # lens = []
        for i in range(0, batch_size):
            doc = []
            sent_count = sent_counts[i]
            sent_len = sent_lens[i]

            for j in range(sent_count):
                # length = sent_len[j]
                cur_sent = last_hidden_states[i, j, 0:1, :]
                # print('cur sent shape', cur_sent.shape)
                doc.append(cur_sent)

            # mean for a doc
            doc_vec = torch.cat(doc, dim=0).unsqueeze(0)
            doc_vec = torch.mean(doc_vec, dim=1)

            # lens.append(doc_vec.shape[0])
            # print(i, 'doc shape', doc_vec.shape)
            docs.append(doc_vec)

        # [batch size, bert embedding dim]
        docs = torch.cat(docs, 0)

        prompt = []
        for j in range(prompt_sent_counts):
            # length = prompt_sent_lens[0][j]
            sent = prompt_hidden_states[j, 0:1, :]
            prompt.append(sent)

        prompt_vec = torch.cat(prompt, dim=0).unsqueeze(0)
        # prompt_vec = self.positional_encoding.forward(prompt_vec)
        # mean [1, bert embedding dim]
        prompt_vec = torch.mean(prompt_vec, dim=1)
        # prompt_vec = self.linear_prompt(prompt_vec)

        doc_feature = docs
        prompt_feature = prompt_vec.expand_as(doc_feature)

        feature = torch.cat([doc_feature, prompt_feature], dim=-1)
        feature = self.dropout_layer(feature)
        log_probs = torch.log_softmax(torch.tanh(self.linear_layer(feature)), dim=-1)

        # log_probs = self.classifier(docs)
        if label is not None:
            loss = self.criterion(input=log_probs.contiguous().view(-1, log_probs.shape[-1]),
                                  target=label.contiguous().view(-1))
        else:
            loss = None

        prediction = torch.max(log_probs, dim=1)[1]
        # print(prediction)
        # print(label)
        return {'loss': loss, 'prediction': prediction}