import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import sys
import yaml
from util.convenient_funcs import create_path, print_model, combine_sentence_to_segment, get_path_prefix, get_feature_from_test_ids
from nltk.tokenize import sent_tokenize
from keras.preprocessing.sequence import pad_sequences
from model.bert_classifier import BertClassifier
from util.data import process_data
from util.make_model import make_model


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

        with open(config['essay_prompt'][set_id]) as f:
            prompt = [f.read()]
            prompt_process = process_data(prompt, tokenizer, True, 300)
            prompt_inputs = prompt_process['inputs']
            prompt_sent_count = prompt_process['sent_count']
            prompt_sent_length = prompt_process['sent_length']
            prompt_mask = prompt_process['attention_mask']

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

        prompt_inputs = torch.tensor(prompt_inputs).to(device)
        prompt_mask = torch.tensor(prompt_mask).to(device)
        prompt_sent_count = torch.tensor(prompt_sent_count).to(device)
        prompt_sent_length = torch.tensor(prompt_sent_length).to(device)

        test_data = TensorDataset(test_inputs, test_masks,
                                  test_sent_count, test_sent_length, test_features)
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
                batch_inputs, batch_masks, batch_sent_count, batch_sent_length, batch_feature = batch

                if 'classifier' in config['model']:
                    result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                   prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, batch_feature,
                                   None)
                else:
                    result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                   prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                                   score_ranges[set_id][0], score_ranges[set_id][1], batch_feature,
                                   None)
                prediction = result['prediction']
                dev_predict.append(prediction)

        dev_predict = torch.cat(dev_predict, dim=0)

        samples = []
        for i in range(0, len(ids)):
            samples.append({})
            samples[i]['domain1_score'] = np.around(dev_predict[i].item())
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
