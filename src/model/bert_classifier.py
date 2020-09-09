import torch
import torch.nn as nn
from transformers import *
from module.cnn_classifier import CNNClassifier
from module.cnn_feature_extracter import CNNFeatureExtrater
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding


class BertClassifier(nn.Module):

    def __init__(self, bert_pretrained_weights, num_class, kernel_size, kernel_nums):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_weights)

        self.positional_encoding = PositionalEncoding(input_dim=768)
        # self.classifier = CNNClassifier(num_class=num_class,
        #                                 input_dim=768,
        #                                 kernel_nums=kernel_nums,
        #                                 kernel_sizes=kernel_size,
        #                                 max_kernel_size=kernel_size[-1])

        # self.essay_feature_extracter = CNNFeatureExtrater(
        #     input_dim=768,
        #     output_dim=300,
        #     kernel_nums=kernel_nums,
        #     kernel_sizes=kernel_size,
        #     max_kernel_size=kernel_size[-1]
        # )
        # self.prompt_feature_extracter = CNNFeatureExtrater(
        #     input_dim=768,
        #     output_dim=300,
        #     kernel_sizes=[2, 4, 8, 16, 32, 64, 128, 256],
        #     kernel_nums=[64, 64, 64, 64, 64, 64, 64, 64],
        #     max_kernel_size=kernel_size[-1]
        # )
        self.linear_layer = nn.Linear(768 * 2, num_class)

        self.dropout_layer = nn.Dropout(0.5)
        self.criterion = nn.NLLLoss(reduction='mean')

    def forward(self, inputs, mask, sent_counts, sent_lens,
                prompt_inputs,
                prompt_mask,
                prompt_sent_counts,
                prompt_sent_lens,
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

        prompt_inputs = prompt_inputs.view(-1, prompt_inputs.shape[-1])
        prompt_mask = prompt_mask.view(-1, prompt_mask.shape[-1])
        prompt_hidden_states = self.bert(input_ids=prompt_inputs, attention_mask=prompt_mask)[0]

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

            doc_vec = torch.cat(doc, dim=0).unsqueeze(0)
            doc_vec = self.positional_encoding.forward(doc_vec)
            doc_vec = torch.mean(doc_vec, dim=1)

            lens.append(doc_vec.shape[0])
            # print(i, 'doc shape', doc_vec.shape)
            docs.append(doc_vec)

        # batch_max_len = max(lens)
        # for i, doc in enumerate(docs):
        #     if doc.shape[0] < batch_max_len:
        #         pd = (0, 0, 0, batch_max_len - doc.shape[0])
        #         m = nn.ConstantPad2d(pd, 0)
        #         doc = m(doc)
        #
        #     docs[i] = doc.unsqueeze(0)

        docs = torch.cat(docs, 0)
        # print(docs.shape)
        # docs = self.positional_encoding.forward(docs)
        # [batch size, num_class]

        prompt = []
        for j in range(prompt_sent_counts):
            length = prompt_sent_lens[0][j]
            sent = prompt_hidden_states[j, :length, :]
            prompt.append(sent)

        prompt_vec = torch.cat(prompt, dim=0).unsqueeze(0)
        prompt_vec = self.positional_encoding.forward(prompt_vec)
        prompt_vec = torch.mean(prompt_vec, dim=1)

        # [batch size, feature size]
        # doc_feature = self.essay_feature_extracter(docs)
        # prompt_feature = self.prompt_feature_extracter(prompt_vec)
        # prompt_feature = prompt_feature.expand_as(doc_feature)

        doc_feature = self.dropout_layer(torch.tanh(docs))
        prompt_feature = self.dropout_layer(torch.tanh(prompt_vec.expand_as(doc_feature)))

        feature = torch.cat([doc_feature, prompt_feature], dim=-1)
        log_probs = torch.log_softmax(self.linear_layer(feature), dim=-1)

        # log_probs = self.classifier(docs)
        if label is not None:
            loss = self.criterion(input=log_probs.contiguous().view(-1, log_probs.shape[-1]),
                                  target=label.contiguous().view(-1))
        else:
            loss = None

        prediction = torch.max(log_probs, dim=1)[1]
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
