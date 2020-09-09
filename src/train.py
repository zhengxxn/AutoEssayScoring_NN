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
from util.convenient_funcs import create_path, print_model, combine_sentence_to_segment, get_path_prefix, get_feature_from_ids
from torch import optim
from util.metrics import kappa
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.data import process_data

from transformers.optimization import AdamW

from util.make_model import make_model


def main():
    torch.manual_seed(0)

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
    # dataset = []
    train_dataset_file = config['train_dataset_file']
    train_dataset = pd.read_csv(train_dataset_file, delimiter='\t', usecols=['essay_set', 'essay_id', 'essay', 'domain1_score'])
    dev_dataset_file = config['dev_dataset_file']
    dev_dataset = pd.read_csv(dev_dataset_file, delimiter='\t', usecols=['essay_set', 'essay_id', 'essay', 'domain1_score'])

    # dataset.append(_train_dataset_)
    # dataset.append(_dev_dataset_)
    # dataset = pd.concat(dataset, axis=0, ignore_index=True)

    essay_set = set(train_dataset['essay_set'])

    # use to save tensor board
    create_path(config['record_path'])
    writer = SummaryWriter(config['record_path'])

    for set_id, essay_set_id in enumerate(essay_set):
        if config['need_training'][set_id] is False:
            continue

        print('begin set ', essay_set_id, 'processing')

        with open(config['essay_prompt'][set_id]) as f:
            prompt = [f.read()]
            prompt_process = process_data(prompt, tokenizer, config['split_segment'], config['segment_max_len'])
            prompt_inputs = prompt_process['inputs']
            prompt_sent_count = prompt_process['sent_count']
            prompt_sent_length = prompt_process['sent_length']
            prompt_mask = prompt_process['attention_mask']

        train_dataset_in_set = train_dataset[train_dataset.essay_set == essay_set_id]
        dev_dataset_in_set = dev_dataset[dev_dataset.essay_set == essay_set_id]

        # essays
        train_essays = train_dataset_in_set.essay.values
        dev_essays = dev_dataset_in_set.essay.values

        # ids
        train_ids = train_dataset_in_set.essay_id.values
        validation_ids = dev_dataset_in_set.essay_id.values

        train_features = get_feature_from_ids(train_ids, config['train_feature'])
        validation_features = get_feature_from_ids(validation_ids, config['validation_feature'])

        train_labels = train_dataset_in_set.domain1_score.values
        validation_labels = dev_dataset_in_set.domain1_score.values

        train_dataset_process = process_data(train_essays, tokenizer, config['split_segment'], config['segment_max_len'])
        dev_dataset_process = process_data(dev_essays, tokenizer, config['split_segment'], config['segment_max_len'])

        train_inputs = train_dataset_process['inputs']
        train_sent_count = train_dataset_process['sent_count']
        train_sent_length = train_dataset_process['sent_length']
        train_masks = train_dataset_process['attention_mask']

        validation_inputs = dev_dataset_process['inputs']
        validation_sent_count = dev_dataset_process['sent_count']
        validation_sent_length = dev_dataset_process['sent_length']
        validation_masks = dev_dataset_process['attention_mask']

        # train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(essay_tokens_pad, labels,
        #                                                                                     random_state=3,
        #                                                                                     test_size=config['dev_dataset_ratio'])
        # train_masks, validation_masks, _, _ = train_test_split(attention_mask, essay_tokens_pad,
        #                                                        random_state=3, test_size=config['dev_dataset_ratio'])
        #
        # train_sent_counts, validation_sent_counts, train_sent_length, validation_sent_length = train_test_split(
        #     essay_sent_count, essay_sent_length, random_state=3, test_size=config['dev_dataset_ratio']
        # )
        # print(train_sent_count)

        train_inputs = torch.tensor(train_inputs).to(device)
        validation_inputs = torch.tensor(validation_inputs).to(device)

        train_labels = torch.tensor(train_labels).to(device)
        validation_labels = torch.tensor(validation_labels).to(device)

        train_masks = torch.tensor(train_masks).to(device)
        validation_masks = torch.tensor(validation_masks).to(device)

        train_sent_counts = torch.tensor(train_sent_count).to(device)
        validation_sent_counts = torch.tensor(validation_sent_count).to(device)

        train_sent_length = torch.tensor(train_sent_length).to(device)
        validation_sent_length = torch.tensor(validation_sent_length).to(device)

        train_features = torch.tensor(train_features).to(device)
        validation_features = torch.tensor(validation_features).to(device)

        prompt_inputs = torch.tensor(prompt_inputs).to(device)
        prompt_mask = torch.tensor(prompt_mask).to(device)
        prompt_sent_count = torch.tensor(prompt_sent_count).to(device)
        prompt_sent_length = torch.tensor(prompt_sent_length).to(device)

        train_data = TensorDataset(train_inputs, train_masks, train_labels,
                                   train_sent_counts, train_sent_length, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['batch_size'][set_id])

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels,
                                        validation_sent_counts, validation_sent_length, validation_features)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=config['batch_size'][set_id])

        print('begin set ', essay_set_id, 'setup model')
        print('model: ', config['model'])
        model = make_model(config, device, set_id)

        # print_model(model)

        # set optimizer only to update the new parameters
        for name, param in model.named_parameters():
            if 'bert' in name \
                    and 'pooler' not in name \
                    and '11' not in name:  # \
                    # and '10' not in name \
                    # and '9' not in name:
                param.requires_grad = False

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=0.00005)

        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True,
        #                                       min_lr=0.00001)

        # begin training
        print('begin set ', essay_set_id, 'begin training')
        create_path(get_path_prefix(config['model_save_path'][set_id]))
        epoch = config['epoch_num'][set_id]
        global_step = 0
        best_validation_kappa = 0
        best_validation_loss = 100000
        for current_epoch in trange(epoch, desc='Epoch'):

            train_loss = []
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.train()
                batch_inputs, batch_masks, batch_labels, batch_sent_count, batch_sent_length, batch_features = batch
                optimizer.zero_grad()

                if 'classifier' in config['model']:
                    result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                   prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, batch_features,
                                   batch_labels)
                else:
                    result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                   prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                                   score_ranges[set_id][0], score_ranges[set_id][1], batch_features,
                                   batch_labels)

                result['loss'].backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)

                train_loss.append(result['loss'].item() / batch_inputs.shape[0])
                # print('loss: ', result['loss'].item())
                optimizer.step()

                global_step += 1

                # evaluation
                # if global_step % 25 == 0:
            dev_true = []
            dev_predict = []
            model.eval()
            dev_loss = []
            with torch.no_grad():
                for batch in validation_dataloader:
                    batch_inputs, batch_masks, batch_labels, batch_sent_count, batch_sent_length, batch_features = batch

                    if 'classifier' in config['model']:
                        result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                       prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, batch_features,
                                       batch_labels)
                    else:
                        result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                       prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                                       score_ranges[set_id][0], score_ranges[set_id][1], batch_features,
                                       batch_labels)

                    prediction = result['prediction']

                    dev_loss.append(result['loss'].item())
                    dev_true.append(batch_labels)
                    dev_predict.append(prediction)

                dev_true = torch.cat(dev_true, dim=0)
                dev_predict = torch.cat(dev_predict, dim=0)

                dev_kappa = kappa(y_true=dev_true, y_pred=dev_predict, weights='quadratic')
                writer.add_scalar(tag='set' + str(essay_set_id) + '_epoch_dev_kappa', scalar_value=dev_kappa,
                                  global_step=current_epoch)

            dev_loss = np.sum(dev_loss) / validation_ids.shape[0]
            writer.add_scalar(tag='set' + str(essay_set_id) + '_epoch_dev_loss', scalar_value=dev_loss,
                              global_step=current_epoch)

            # if dev_loss < best_validation_loss:
            #     print('get better result save')
            #     best_validation_loss = dev_loss
            #     # best_validation_kappa = dev_kappa
            #     torch.save(model.state_dict(), config['model_save_path'][set_id])

            if dev_kappa > best_validation_kappa:
                print('get better kappa result, save')
                best_validation_kappa = dev_kappa
                torch.save(model.state_dict(), config['model_save_path'][set_id])

            # lr_scheduler.step(np.average(dev_loss))
            print('dev_kappa is', dev_kappa)
            print('dev loss ', dev_loss)

            writer.add_scalar('set'+str(essay_set_id)+'_epoch_avg_train_loss', scalar_value=np.average(train_loss), global_step=current_epoch)
            print('average train loss: ', np.average(train_loss))
            print()

        del model
        del train_inputs
        del validation_inputs
        del train_masks
        del validation_masks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()