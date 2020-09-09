from model.bert_simple_classifier import BertSimpleClassifier
from model.bert_simple_regressor import BertSimpleRegressor
from model.bert_attention_classifier import BertAttentionClassifier
from model.bert_attention_regressor import BertAttentionRegressor
from model.bert_global_attention_classifier import BertGlobalAttentionClassifier
from model.bert_global_attention_regressor import BertGlobalAttentionRegressor
from model.bert_single_classifier import BertSingleClassifier
from model.bert_cnn_classifier import BertCNNClassifier
from model.bert_most_single_classifier import BertMostSingleClassifier
from model.bert_recurrent_attention_regressor import BertRecurrentAttentionRegressor
from model.bert_recurrent_attention_classifier import BertRecurrentAttentionClassifier
from model.bert_recurrent_attention_regressor_mix import MixBertRecurrentAttentionRegressor
from model.bert_regressor_with_domain_classifier import MixBertRegressorWithDomainClassifier


def make_model(config, device, set_id):
    model_name = config['model']
    model = None
    if model_name == 'bert_simple_classifier':
        model = BertSimpleClassifier(config['bert_model_path'],
                                     num_class=config['score_ranges'][set_id][1] + 1,
                                     ).to(device)

    elif model_name == 'bert_simple_regressor':
        model = BertSimpleRegressor(config['bert_model_path']).to(device)

    elif model_name == 'bert_attention_classifier':
        model = BertAttentionClassifier(config['bert_model_path'],
                                        num_class=config['score_ranges'][set_id][1] + 1,
                                        ).to(device)

    elif model_name == 'bert_attention_regressor':
        model = BertAttentionRegressor(config['bert_model_path']).to(device)

    elif model_name == 'bert_global_attention_classifier':
        model = BertGlobalAttentionClassifier(config['bert_model_path'],
                                              num_class=config['score_ranges'][set_id][1] + 1,
                                     ).to(device)

    elif model_name == 'bert_global_attention_regressor':
        model = BertGlobalAttentionRegressor(config['bert_model_path']).to(device)

    elif model_name == 'bert_single_classifier':
        model = BertSingleClassifier(config['bert_model_path'],
                                     num_class=config['score_ranges'][set_id][1] + 1,
                                     ).to(device)

    elif model_name == 'bert_cnn_classifier':
        model = BertCNNClassifier(config['bert_model_path'],
                                  num_class=config['score_ranges'][set_id][1] + 1,
                                  kernel_nums=config['kernel_num'],
                                  kernel_size=config['kernel_size']).to(device)

    elif model_name == 'bert_most_single_classifier':
        model = BertMostSingleClassifier(config['bert_model_path'],
                                         num_class=config['score_ranges'][set_id][1] + 1,
                                         ).to(device)

    elif model_name == 'bert_recurrent_attention_regressor':
        model = BertRecurrentAttentionRegressor(config['bert_model_path']).to(device)

    elif model_name == 'bert_recurrent_attention_classifier':
        model = BertRecurrentAttentionClassifier(config['bert_model_path'],
                                         num_class=config['score_ranges'][set_id][1] + 1,
                                         ).to(device)

    elif model_name == 'mix_bert_recurrent_attention_regressor':
        model = MixBertRecurrentAttentionRegressor(config['bert_model_path']).to(device)

    elif model_name == 'mix_bert_regressor_with_domain_classifier':
        model = MixBertRegressorWithDomainClassifier(config['bert_model_path'], len(config['used_set'][set_id])).to(device)

    return model
