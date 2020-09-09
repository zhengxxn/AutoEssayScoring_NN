import torch
import torch.nn as nn
from transformers import *
from module.cnn_regressor import CNNRegressor
from module.cnn_feature_extracter import CNNFeatureExtrater


class BertRegressor(nn.Module):

    def __init__(self, bert_pretrained_weights, kernel_size, kernel_nums, min_score, max_score):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_weights)

        self.essay_feature_extracter = CNNFeatureExtrater(
            input_dim=768,
            output_dim=300,
            kernel_nums=kernel_nums,
            kernel_sizes=kernel_size,
            max_kernel_size=kernel_size[-1]
        )
        self.prompt_feature_extracter = CNNFeatureExtrater(
            input_dim=768,
            output_dim=300,
            kernel_sizes=[2, 4, 8, 16, 32, 64, 128, 256],
            kernel_nums=[64, 64, 64, 64, 64, 64, 64, 64],
            max_kernel_size=kernel_size[-1]
        )
        self.linear_layer = nn.Linear(300 * 2, 1)

        self.criterion = nn.MSELoss(reduction='mean')

        self.min_score = min_score
        self.max_score = max_score

    def forward(self, inputs, mask, sent_counts, sent_lens, label=None):
        """

        :param max_score:
        :param min_score:
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

        docs = []
        lens = []
        for i in range(0, batch_size):
            doc = []
            sent_count = sent_counts[i]
            sent_len = sent_lens[i]

            for j in range(sent_count):
                length = sent_len[j]
                cur_sent = last_hidden_states[i, j, :length, :]
                # print('cur sent shape', cur_sent.shape)
                doc.append(cur_sent)

            doc_vec = torch.cat(doc, dim=0)
            lens.append(doc_vec.shape[0])
            # print(i, 'doc shape', doc_vec.shape)
            docs.append(doc_vec)

        batch_max_len = max(lens)
        for i, doc in enumerate(docs):
            if doc.shape[0] < batch_max_len:
                pd = (0, 0, 0, batch_max_len - doc.shape[0])
                m = nn.ConstantPad2d(pd, 0)
                doc = m(doc)

            docs[i] = doc.unsqueeze(0)

        # for doc in docs:
        #     print(doc.shape)

        docs = torch.cat(docs, 0)
        # print(docs.shape)

        # [batch size, num_class]
        logit = self.regressor(docs)  # 0 ~ 1
        # logit = self.min_score + logit * (self.max_score - self.min_score)

        if label is not None:
            label = (label - self.min_score) / (self.max_score - self.min_score)
            loss = self.criterion(input=logit.contiguous().view(-1),
                                  target=label.type_as(logit).contiguous().view(-1))
        else:
            loss = None

        prediction = logit * (self.max_score - self.min_score) + self.min_score
        # prediction = logit  # torch.max(log_probs, dim=1)[1]
        return {'loss': loss, 'prediction': prediction}
    # def forward(self, input_tokens, label=None):
    #
    #     last_hidden_states = self.bert(input_tokens)
    #
    #     logit = self.classifier(last_hidden_states)
    #
    #     if label is not  None:
    #         loss = self.criterion(logit, label)
    #         return loss
    #
    #     else:
    #         return logit
