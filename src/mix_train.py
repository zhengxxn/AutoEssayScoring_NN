import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import sys
import yaml
from util.convenient_funcs import create_path, print_model, combine_sentence_to_segment, get_path_prefix, \
    get_feature_from_ids
from torch import optim
from util.metrics import kappa
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util.data import process_data
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
    train_dataset = pd.read_csv(train_dataset_file, delimiter='\t',
                                usecols=['essay_set', 'essay_id', 'essay', 'domain1_score'])
    dev_dataset_file = config['dev_dataset_file']
    dev_dataset = pd.read_csv(dev_dataset_file, delimiter='\t',
                              usecols=['essay_set', 'essay_id', 'essay', 'domain1_score'])

    essay_set = set(train_dataset['essay_set'])

    # use to save tensor board
    create_path(config['record_path'])
    writer = SummaryWriter(config['record_path'])

    for set_id, essay_set_id in enumerate(essay_set):
        if config['need_training'][set_id] is False:
            continue

        print('begin set ', essay_set_id, 'processing')

        # prepare prompt
        train_essays = []
        validation_essays = []
        train_ids = []
        validation_ids = []
        train_features = []
        validation_features = []
        train_labels = []
        validation_labels = []
        train_prompt_essays = []
        validation_prompt_essays = []
        train_max_scores = []
        train_min_scores = []
        validation_max_scores = []
        validation_min_scores = []
        train_domain_label = []
        validation_domain_label = []

        # train_avg_len = []
        # validation_avg_len = []
        # train_score_bias = []
        # validation_score_bias = []

        current_domain = 0
        for current_set_id, current_essay_set_id in enumerate(essay_set):
            if current_essay_set_id not in config['used_set'][set_id]:
            # if current_set_id == set_id:
                continue
            else:
                print('used set', current_essay_set_id)

            current_train_dataset_in_set = train_dataset[train_dataset.essay_set == current_essay_set_id]
            current_validation_dataset_int_set = dev_dataset[dev_dataset.essay_set == current_essay_set_id]

            current_train_essays = current_train_dataset_in_set.essay.values
            train_essays.extend(current_train_essays)
            current_validation_essays = current_validation_dataset_int_set.essay.values
            validation_essays.extend(current_validation_essays)

            current_train_ids = current_train_dataset_in_set.essay_id.values
            train_ids.extend(current_train_ids)
            current_validation_ids = current_validation_dataset_int_set.essay_id.values
            validation_ids.extend(current_validation_ids)

            current_train_features = get_feature_from_ids(current_train_ids, config['train_feature'])
            train_features.extend(current_train_features)
            current_validation_features = get_feature_from_ids(current_validation_ids, config['validation_feature'])
            validation_features.extend(current_validation_features)

            current_train_labels = current_train_dataset_in_set.domain1_score.values
            train_labels.extend(current_train_labels)
            current_validation_labels = current_validation_dataset_int_set.domain1_score.values
            validation_labels.extend(current_validation_labels)

            current_train_max_scores = [config['score_ranges'][current_set_id][1]] * len(current_train_ids)
            current_train_min_scores = [config['score_ranges'][current_set_id][0]] * len(current_train_ids)
            current_validation_max_scores = [config['score_ranges'][current_set_id][1]] * len(current_validation_ids)
            current_validation_min_scores = [config['score_ranges'][current_set_id][0]] * len(current_validation_ids)
            train_max_scores.extend(current_train_max_scores)
            train_min_scores.extend(current_train_min_scores)
            validation_max_scores.extend(current_validation_max_scores)
            validation_min_scores.extend(current_validation_min_scores)

            # current_train_avg_len = [config['avg_length'][current_set_id]] * len(current_train_ids)
            # current_validation_avg_len = [config['avg_length'][current_set_id]] * len(current_validation_ids)
            # train_avg_len.extend(current_train_avg_len)
            # validation_avg_len.extend(current_validation_avg_len)

            # current_train_score_bias = [config['score_bias'][current_set_id]] * len(current_train_ids)
            # current_validation_score_bias = [config['score_bias'][current_set_id]] * len(current_validation_ids)
            # train_score_bias.extend(current_train_score_bias)
            # validation_score_bias.extend(current_validation_score_bias)

            current_train_domain_label = [current_domain] * len(current_train_ids)
            current_validation_domain_label = [current_domain] * len(current_validation_ids)
            train_domain_label.extend(current_train_domain_label)
            validation_domain_label.extend(current_validation_domain_label)
            current_domain += 1

            with open(config['essay_prompt'][set_id]) as f:
                current_prompt_essays = [f.read()]
            current_train_prompt_essays = current_prompt_essays * len(current_train_ids)
            train_prompt_essays.extend(current_train_prompt_essays)
            current_validation_prompt_essays = current_prompt_essays * len(current_validation_ids)
            validation_prompt_essays.extend(current_validation_prompt_essays)

        train_prompt_process = process_data(train_prompt_essays, tokenizer, config['split_segment'],
                                            config['segment_max_len'])
        validation_prompt_process = process_data(validation_prompt_essays, tokenizer, config['split_segment'],
                                                 config['segment_max_len'])
        train_prompt_inputs = train_prompt_process['inputs']
        train_prompt_sent_count = train_prompt_process['sent_count']
        train_prompt_sent_length = train_prompt_process['sent_length']
        train_prompt_mask = train_prompt_process['attention_mask']

        validation_prompt_inputs = validation_prompt_process['inputs']
        validation_prompt_sent_count = validation_prompt_process['sent_count']
        validation_prompt_sent_length = validation_prompt_process['sent_length']
        validation_prompt_mask = validation_prompt_process['attention_mask']

        train_dataset_process = process_data(train_essays, tokenizer, config['split_segment'],
                                             config['segment_max_len'])
        dev_dataset_process = process_data(validation_essays, tokenizer, config['split_segment'],
                                           config['segment_max_len'])

        train_inputs = train_dataset_process['inputs']
        train_sent_count = train_dataset_process['sent_count']
        train_sent_length = train_dataset_process['sent_length']
        train_masks = train_dataset_process['attention_mask']

        validation_inputs = dev_dataset_process['inputs']
        validation_sent_count = dev_dataset_process['sent_count']
        validation_sent_length = dev_dataset_process['sent_length']
        validation_masks = dev_dataset_process['attention_mask']

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

        train_max_scores = torch.tensor(train_max_scores).to(device)
        train_min_scores = torch.tensor(train_min_scores).to(device)
        validation_max_scores = torch.tensor(validation_max_scores).to(device)
        validation_min_scores = torch.tensor(validation_min_scores).to(device)

        train_domain_label = torch.tensor(train_domain_label).to(device)
        validation_domain_label = torch.tensor(validation_domain_label).to(device)

        train_prompt_inputs = torch.tensor(train_prompt_inputs).to(device)
        train_prompt_mask = torch.tensor(train_prompt_mask).to(device)
        train_prompt_sent_count = torch.tensor(train_prompt_sent_count).to(device)
        train_prompt_sent_length = torch.tensor(train_prompt_sent_length).to(device)

        validation_prompt_inputs = torch.tensor(validation_prompt_inputs).to(device)
        validation_prompt_mask = torch.tensor(validation_prompt_mask).to(device)
        validation_prompt_sent_count = torch.tensor(validation_prompt_sent_count).to(device)
        validation_prompt_sent_length = torch.tensor(validation_prompt_sent_length).to(device)

        # train_avg_len = torch.tensor(train_avg_len).to(device)
        # validation_avg_len = torch.tensor(validation_avg_len).to(device)
        # train_score_bias = torch.tensor(train_score_bias).to(device)
        # validation_score_bias = torch.tensor(validation_score_bias).to(device)

        train_data = TensorDataset(train_inputs, train_masks, train_labels,
                                   train_sent_counts, train_sent_length, train_features,
                                   train_prompt_inputs, train_prompt_mask, train_prompt_sent_count,
                                   train_prompt_sent_length,
                                   train_max_scores, train_min_scores,
                                   train_domain_label, )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['batch_size'][set_id])

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels,
                                        validation_sent_counts, validation_sent_length, validation_features,
                                        validation_prompt_inputs, validation_prompt_mask, validation_prompt_sent_count,
                                        validation_prompt_sent_length,
                                        validation_max_scores, validation_min_scores,
                                        validation_domain_label, )

        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler,
                                           batch_size=config['batch_size'][set_id])

        print('begin set ', essay_set_id, 'setup model')
        print('model: ', config['model'])
        model = make_model(config, device, set_id)

        # print_model(model)

        # set optimizer only to update the new parameters
        for name, param in model.named_parameters():
            if 'bert' in name \
                    and 'pooler' not in name \
                    and '11' not in name:
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
        # best_validation_kappa = 0
        best_validation_loss = 100000
        for current_epoch in trange(epoch, desc='Epoch'):

            train_loss = []
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.train()
                batch_inputs, batch_masks, batch_labels, batch_sent_count, batch_sent_length, batch_features, \
                prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, max_scores, min_scores, \
                train_domain_label = batch

                optimizer.zero_grad()

                total_batch_count = int(train_inputs.shape[0] / batch_inputs.shape[0])
                p = (step + current_epoch * total_batch_count) / epoch / total_batch_count
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                               prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                               min_scores, max_scores, batch_features,
                               batch_labels, domain_label=train_domain_label, alpha=alpha)

                result['loss'].backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)

                train_loss.append(result['loss'].item())
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
                    batch_inputs, batch_masks, batch_labels, batch_sent_count, batch_sent_length, batch_features, \
                    prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length, max_scores, min_scores, \
                    domain_label = batch

                    result = model(batch_inputs, batch_masks, batch_sent_count, batch_sent_length,
                                   prompt_inputs, prompt_mask, prompt_sent_count, prompt_sent_length,
                                   min_scores, max_scores, batch_features,
                                   batch_labels, domain_label=None, alpha=None)

                    # prediction = result['prediction']

                    dev_loss.append(result['loss'].item())
                    # dev_true.append(batch_labels)
                    # dev_predict.append(prediction)

                # dev_true = torch.cat(dev_true, dim=0)
                # dev_predict = torch.cat(dev_predict, dim=0)

                # dev_kappa = kappa(y_true=dev_true, y_pred=dev_predict, weights='quadratic')
                # writer.add_scalar(tag='set' + str(essay_set_id) + '_epoch_dev_kappa', scalar_value=dev_kappa,
                #                   global_step=current_epoch)
            dev_loss = np.sum(dev_loss) / len(validation_ids)
            writer.add_scalar(tag='set' + str(essay_set_id) + '_epoch_dev_loss', scalar_value=dev_loss,
                              global_step=current_epoch)

            if dev_loss < best_validation_loss:
                print('get better result save')
                best_validation_loss = dev_loss
                # best_validation_kappa = dev_kappa
                torch.save(model.state_dict(), config['model_save_path'][set_id])

            # lr_scheduler.step(np.average(dev_loss))
            # print('dev_kappa is', dev_kappa)
            print('dev loss ', dev_loss)

            writer.add_scalar('set' + str(essay_set_id) + '_epoch_avg_train_loss', scalar_value=np.sum(train_loss) / len(train_ids),
                              global_step=current_epoch)
            print('average train loss: ', np.sum(train_loss) / len(train_ids))
            print()

        del model
        del train_inputs
        del validation_inputs
        del train_masks
        del validation_masks
        del train_prompt_inputs
        del train_prompt_mask
        del validation_prompt_inputs
        del validation_prompt_mask
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
