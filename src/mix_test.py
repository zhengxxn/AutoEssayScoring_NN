import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import sys
import yaml
from util.convenient_funcs import create_path, print_model, get_path_prefix, get_feature_from_test_ids, more_uniform
from util.data import process_data
from util.make_model import make_model
import math


def main():
    print('cuda is available: ', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_file_path = sys.argv[1]

    print('read config')
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    score_ranges = config['score_ranges']

    bert_model_path = config['bert_model_path']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    print('read dataset')

    test_dataset_file = config['test_dataset_file']
    test_dataset = pd.read_csv(test_dataset_file, delimiter='\t', usecols=['essay_set', 'essay_id', 'essay'])

    essay_set = set(test_dataset['essay_set'])

    for set_id, essay_set_id in enumerate(essay_set):
        if config['need_test'][set_id] is False:
            continue

        print('begin set ', essay_set_id, 'processing')

        test_dataset_in_set = test_dataset[test_dataset.essay_set == essay_set_id]

        test_essays = test_dataset_in_set.essay.values

        test_dataset_process = process_data(test_essays, tokenizer, config['split_segment'], config['segment_max_len'])

        ids = test_dataset_in_set.essay_id.values
        test_features = get_feature_from_test_ids(ids, config['test_feature'])

        test_inputs = test_dataset_process['inputs']
        test_sent_count = test_dataset_process['sent_count']
        test_sent_length = test_dataset_process['sent_length']
        test_masks = test_dataset_process['attention_mask']

        test_inputs = torch.tensor(test_inputs).to(device)
        # test_labels = torch.tensor(test_labels).to(device)
        test_masks = torch.tensor(test_masks).to(device)
        test_sent_count = torch.tensor(test_sent_count).to(device)
        test_sent_length = torch.tensor(test_sent_length).to(device)
        test_features = torch.tensor(test_features).to(device)

        with open(config['essay_prompt'][set_id]) as f:
            test_prompt = [f.read()] * len(ids)
            test_prompt_process = process_data(test_prompt, tokenizer, config['split_segment'], config['segment_max_len'])
            test_prompt_inputs = test_prompt_process['inputs']
            test_prompt_sent_count = test_prompt_process['sent_count']
            test_prompt_sent_length = test_prompt_process['sent_length']
            test_prompt_mask = test_prompt_process['attention_mask']

        test_prompt_inputs = torch.tensor(test_prompt_inputs).to(device)
        test_prompt_mask = torch.tensor(test_prompt_mask).to(device)
        test_prompt_sent_count = torch.tensor(test_prompt_sent_count).to(device)
        test_prompt_sent_length = torch.tensor(test_prompt_sent_length).to(device)

        test_max_scores = [config['score_ranges'][set_id][1]] * len(ids)
        test_min_scores = [config['score_ranges'][set_id][0]] * len(ids)
        test_max_scores = torch.tensor(test_max_scores).to(device)
        test_min_scores = torch.tensor(test_min_scores).to(device)

        test_data = TensorDataset(test_inputs, test_masks, test_sent_count, test_sent_length, test_features,
                                  test_prompt_inputs, test_prompt_mask, test_prompt_sent_count, test_prompt_sent_length,
                                  test_max_scores, test_min_scores)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler,
                                     batch_size=config['batch_size'][set_id])

        print('begin set ', essay_set_id, 'setup model')
        model = make_model(config, device, set_id)

        # set optimizer only to update the new parameters
        model.load_state_dict(torch.load(config['model_save_path'][set_id]))

        # begin training
        print('begin set ', essay_set_id, 'begin test')

        # evaluation
        model.eval()
        with torch.no_grad():
            dev_predict = []
            for batch in test_dataloader:
                batch_inputs, batch_masks, batch_sent_count, batch_sent_length, batch_feature, \
                    prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, \
                    batch_max_scores, batch_min_scores = batch

                result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                               prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                               batch_min_scores, batch_max_scores, batch_feature,
                               None, None, None)
                prediction = result['prediction']
                prediction = prediction[:, 0]
                dev_predict.append(prediction)

        dev_predict = torch.cat(dev_predict, dim=0)
        dev_predict = dev_predict.tolist()

        predict_average = np.average(dev_predict)
        gap = config['mean_score'][set_id] - predict_average

        # predict_average = np.average(dev_predict)
        # gap = config['mean_score'][set_id] - predict_average
        #
        # if essay_set_id in [1, 2, 7, 8]:
        #     if gap < 0:
        #         gap = -math.pow(-gap, 0.666)
        #     else:
        #         gap = math.pow(gap, 0.666)
        #
        # #
        # dev_predict = [temp + gap for temp in dev_predict]
        # if essay_set_id in [2, 3, 4, 5, 6]:
        #     dev_predict = more_uniform(dev_predict)

        # dev_predict = [temp if temp > score_ranges[set_id][0] else score_ranges[set_id][0] for temp in dev_predict]
        # dev_predict = [temp if temp < score_ranges[set_id][1] else score_ranges[set_id][1] for temp in dev_predict]

        samples = []
        for i in range(0, len(ids)):
            samples.append({})
            samples[i]['domain1_score'] = dev_predict[i]  # np.around(dev_predict[i])  # np.around(dev_predict[i].item())
            samples[i]['essay_id'] = ids[i]
            samples[i]['essay_set'] = essay_set_id
        create_path(get_path_prefix(config['test_output_path'][set_id]))
        save_to_tsv(samples, config['test_output_path'][set_id])

        del model
        del test_inputs
        del test_masks
        torch.cuda.empty_cache()


def save_to_tsv(samples: list, tsv_file):
    raw_data = {
        'id': [sample['essay_id'] for sample in samples],
        'set': [sample['essay_set'] for sample in samples],
        'score': [sample['domain1_score'] for sample in samples]
    }
    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file, sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
