import pandas as pd
from util.metrics import kappa
import numpy as np
from scipy.stats import normaltest
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statistics
from scipy.stats import wasserstein_distance


dev_results = [
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/1.tsv',
    '/Users/zx/Documents/mix_uncased_with_classifier_selected/2.tsv',
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/3.tsv',
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/4.tsv',
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/5.tsv',
    '/Users/zx/Documents/mix_uncased_with_classifier_selected/6.tsv',
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/7.tsv',
    '/Users/zx/Documents/mix_uncased_bertRecurrentAttentionRegressor_dev/8.tsv',
]

# mean_value = [8.0, 3.0, 1.5, 1.0, 2.0, 2.0, 16.0, 36.0]
# mean_value = [7.9, 3.3, 0, 0, 0, 0, 17.5, 33.0]
score_ranges = [
    [2, 12],
    [1, 6],
    [0, 3],
    [0, 3],
    [0, 4],
    [0, 4],
    [0, 30],
    [0, 60],
]

train_dataset_file = '/Users/zx/Documents/课程/文章自动评分/essay_data/train.tsv'
dev_dataset_file = '/Users/zx/Documents/课程/文章自动评分/essay_data/dev.tsv'
train_dataset = pd.read_csv(train_dataset_file, delimiter='\t',
                          usecols=['essay_set', 'essay_id', 'domain1_score'])
dev_dataset = pd.read_csv(dev_dataset_file, delimiter='\t',
                          usecols=['essay_set', 'essay_id', 'domain1_score'])

k_s = []
dev_true_s = []
train_true_s = []
for i in range(6, 7):
    # if i in [3, 4, 5, 6]:
    #     continue
    dev_in_set = dev_dataset[dev_dataset.essay_set == i]
    train_in_set = train_dataset[train_dataset.essay_set == i]
    # print(dev_in_set.domain1_score.values)

    with open(dev_results[i - 1], 'r') as f:
        ls = f.readlines()
        dev_predict = [float(l.split('\t')[-1][:-1]) for l in ls]

    # print('median', statistics.median(dev_predict))
    # print('average ', np.average(dev_predict))
    # gap = mean_value[i - 1] - np.average(dev_predict)
    # gap = gap * 0.618
    # print(i, 'gap', gap)
    # if gap < 0:
    #     if i in [1, 2, 7, 8]:
    #         gap = -math.pow(-gap, 0.666)
    # else:
    #     if i in [1, 2, 7, 8]:
    #         gap = math.pow(gap, 0.666)
    # # #
    # if i in [1, 2, 7, 8]:
        # print(i, 'gap', gap)
        # dev_predict = [temp + gap for temp in dev_predict]
    # if i in [1, 2, 3, 4, 5, 6, 7, 8]:
    #     dev_predict = more_uniform(dev_predict)
        #     dev_predict = np.around(dev_predict)
            # sns.distplot(dev_predict, color='g', norm_hist=True)
            # plt.show()

    #     print('average', np.average(dev_predict))
    dev_predict = [temp if temp > score_ranges[i - 1][0] else score_ranges[i - 1][0] for temp in dev_predict]
    dev_predict = [temp if temp < score_ranges[i - 1][1] else score_ranges[i - 1][1] for temp in dev_predict]
    # print('max ', max(dev_predict))
    # print('min ', min(dev_predict))
    #
    print(dev_predict)
    dev_true = dev_in_set.domain1_score.values
    print(np.average(dev_true))
    # dev_true = [(v - score_ranges[i-1][0]) / (score_ranges[i-1][1] - score_ranges[i-1][0]) for v in dev_true]
    # dev_true_s.append(dev_true)
    #
    # train_true = train_in_set.domain1_score.values
    # train_true = [(v - score_ranges[i-1][0]) / (score_ranges[i-1][1] - score_ranges[i-1][0]) for v in train_true]
    # train_true_s.append(train_true)
    # sns.distplot(dev_true, color='b', norm_hist=True)
    # plt.show()
    # print(dev_true)
    # print('true max ', max(dev_true))
    # print('true min ', min(dev_true))
    k = kappa(y_true=dev_true, y_pred=dev_predict, weights='quadratic')
    # k_s.append(k)
    print('kappa', k)

for dev_true in dev_true_s:
    distances = []
    for train_true in train_true_s:
        distances.append(wasserstein_distance(dev_true, train_true))
    print(distances)

# print(np.average(k_s))

# set1_pseudo_mean = 7.0 + (((16.0 - 15) / 15 + (36.0 - 30) / 30) / 2) * 7.0
# print(set1_pseudo_mean)
#
# set7_pseudo_mean = 15.0 + ((8.0 - 7.0) / 7.0 + (36.0 - 30.0) / 30) / 2 * 15.0
# print(set7_pseudo_mean)
#
# set8_pseudo_mean = 30.0 + ((8.0 - 7.0) / 7.0 + (16.0 - 15.0) / 15.0) / 2 * 30.0
# print(set8_pseudo_mean)
#
# # 2, 3, 4, 5, 6
#
# set2_pseudo_mean = 3.5 + ((1.0-1.5) / 1.5 * 2) / 4
# print(set2_pseudo_mean)
#
# set3_pseudo_mean = 1.5 + ((3.0-3.5) / 3.0 + (1.0 - 1.5) / 1.5) / 4
# print(set3_pseudo_mean)
#
# set4_pseudo_mean = 1.5 + ((3.0-3.5) / 3.0 + (1.0 - 1.5) / 1.5) / 4
# print(set4_pseudo_mean)
#
# set5_pseudo_mean = 2.0 + ((3.0-3.5) / 3.0 + (1.0 - 1.5) / 1.5 * 2) / 4
# print(set5_pseudo_mean)
#
# set6_pseudo_mean = 2.0 + ((3.0-3.5) / 3.0 + (1.0 - 1.5) / 1.5 * 2) / 4
# print(set6_pseudo_mean)