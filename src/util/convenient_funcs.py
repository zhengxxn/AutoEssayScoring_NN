import pandas as pd
# from sklearn.model_selection import train_test_split
import re
from pathlib import Path
from collections import Counter
import pickle
import numpy as np


def tensor2str(prediction, vocab):
    str = []

    for i in range(0, prediction.size(0)):

        ch = vocab.itos[prediction[i]]

        if ch == '<eos>':
            break
        else:
            str.append(ch)

    return " ".join(str)


def convert_xml_to_plaintext(src_file, trg_file):
    with open(src_file, 'r') as f:
        with open(trg_file, 'w') as wf:
            newlines = []
            lines = f.readlines()
            for (i, line) in enumerate(lines):
                newline = re.sub('<seg id=\"[0-9]+\"> | </seg>', '', line, 2)
                if '<' not in newline:
                    newlines.append(newline)

            wf.writelines(newlines)


def save_to_tsv(file_path_1, file_path_2, tsv_file_path, domain=None):

    with open(file_path_1, encoding='utf-8') as f:
        src = f.read().split('\n')[:-1]
    with open(file_path_2, encoding='utf-8') as f:
        trg = f.read().split('\n')[:-1]

    if domain is not None:
        raw_data = {'src': [line for line in src], 'trg': [line for line in trg], 'domain': [domain for line in src]}
    else:
        raw_data = {'src': [line for line in src], 'trg': [line for line in trg]}
    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file_path, index=False, sep='\t')


def new_save_to_tsv(config, tsv_file_path):

    raw_data = {}
    for key in config.keys():
        file_name = config[key]
        with open(file_name, encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]
            value = [line for line in lines]
            raw_data[key] = value

    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file_path, index=False, sep='\t')


def get_path_prefix(path):
    return re.sub('/[^/]+$', '', path, 1)


def create_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def de_bpe(str):
    return re.sub(r'@@ |@@ ?$', '', str)


def generate_vocab_counter(file):
    c = Counter()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word, freq = line.split(' ')
            c[word] = int(freq)

    return c


def print_model(model):
    print(model)
    for name, param in model.named_parameters():
        print(name, param.size())


def combine_sentence_to_segment(sents: list, max_segment_len=400):
    segments = []

    segment = ''
    for sent in sents:
        if segment == '':
            segment = sent
        elif len(segment.split(' ')) + len(sent.split(' ')) > max_segment_len:
            segments.append(segment)
            segment = ''
            segment = segment + sent
        else:
            segment = segment + ' ' + sent

    segments.append(segment)
    return segments


def get_feature_from_ids(ids, file_name):
    with open('/home/user_data55/zhengx/project/data/auto_score/train.feature', 'rb') as train_f:
        train_features = {}
        train_features = pickle.load(train_f)

    with open('/home/user_data55/zhengx/project/data/auto_score/dev.feature', 'rb') as dev_f:
        dev_featues = {}
        dev_featues = pickle.load(dev_f)

    features = []
    for id in ids:
        if id in train_features.keys():
            features.append(train_features[id])
        else:
            features.append(dev_featues[id])

    return features


def get_feature_from_test_ids(ids, filename):
    with open('/home/user_data55/zhengx/project/data/auto_score/test.feature', 'rb') as test_f:
        test_features = {}
        test_features = pickle.load(test_f)

    features = []
    for id in ids:
        if id in test_features.keys():
            features.append(test_features[id])
        else:
            features.append(test_features[id])

    return features


def more_uniform(values):
    mean = np.average(values)
    for i, value in enumerate(values):
        gap = value - mean
        if 0 < gap < 1:
            value += 0.5
        if 0 > gap > 1:
            value -= 0.5
        values[i] = value

    return values