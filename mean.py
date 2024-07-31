import csv
import itertools
import logging
import os

import numpy as np
import pandas as pd

import fold
from dataset import Dataset
from evaluate import TopK


class Mean:
    def __init__(self, folds: list):
        self.folds = folds
        self.dataframes = {
            'means': self.means(),
            'tops': self.top(),
            'true_positive': self.true_positive(),
        }
        self.dataframes.update({'best': self.best(self.dataframes['means'])})

    def mean_std(self, column, df):
        mean = df.groupby(column).mean().reset_index()
        std = df.groupby(column).std().reset_index()
        return mean, std

    def means(self):
        evals = pd.concat(fold.dataframes['evals'] for fold in self.folds)
        mean, std = self.mean_std('rule', evals)
        mean.rename(columns={'accuracy': 'mean_accuracy', 'f1': 'mean_f1'}, inplace=True)
        std.rename(columns={'accuracy': 'std_accuracy', 'f1': 'std_f1'}, inplace=True)
        return mean.merge(std, how='inner', on='rule')

    def best(self, means):
        df_accuracy = means.loc[means['mean_accuracy'].idxmax()]
        df_f1 = means.loc[means['mean_f1'].idxmax()]
        data = {'metric': ['mean_accuracy', 'mean_f1', 'std_accuracy', 'std_f1'],
                'value': [df_accuracy['mean_accuracy'], df_f1['mean_f1'], df_accuracy['std_accuracy'], df_f1['std_f1']],
                'rule': [df_accuracy['rule'], df_f1['rule'], df_accuracy['rule'], df_f1['rule']]}
        return pd.DataFrame(data, columns=list(data.keys()))

    def mean_std_columns(self, column, columns, df):
        mean = df.groupby(columns)[column].mean().reset_index()
        std = df.groupby(columns)[column].std().reset_index()
        mean.rename(columns={column: 'mean_%s' % column}, inplace=True)
        std.rename(columns={column: 'std_%s' % column}, inplace=True)
        return mean, std

    def top(self):
        tops = pd.concat(fold.dataframes['tops'] for fold in self.folds)
        mean_top, std_top = self.mean_std_columns('topk_accuracy_score', ['k', 'rule'], tops)
        mean_count_test, std_count_test = self.mean_std_columns('count_test', ['k', 'rule'], tops)

        top = mean_top.merge(std_top, how='inner', on=['rule', 'k'])
        count_test = mean_count_test.merge(std_count_test, how='inner', on=['rule', 'k'])
        top_count_test = top.merge(count_test, how='inner', on=['rule', 'k'])
        top_count_test['mean_topk_accuracy_score+100'] = top_count_test['mean_topk_accuracy_score'] / top_count_test[
            'mean_count_test']
        return top_count_test

    def true_positive(self):
        true_positive = pd.concat(fold.dataframes['true_positive'] for fold in self.folds)
        mean_tps, std_tps = self.mean_std_columns('true_positive', ['rule', 'labels'], true_positive)
        mean_count_train, std_count_train = self.mean_std_columns('count_train', ['rule', 'labels'], true_positive)
        mean_count_test, std_count_test = self.mean_std_columns('count_test', ['rule', 'labels'], true_positive)

        tps = mean_tps.merge(std_tps, how='inner', on=['rule', 'labels'])
        count_train = mean_count_train.merge(std_count_train, how='inner', on=['rule', 'labels'])
        count_test = mean_count_test.merge(std_count_test, how='inner', on=['rule', 'labels'])
        tps_count_train = tps.merge(count_train, how='inner', on=['rule', 'labels'])
        tps_count_train_count_test = tps_count_train.merge(count_test, how='inner', on=['rule', 'labels'])
        return tps_count_train_count_test

    def save(self, output):
        output = os.path.join(output, 'mean')
        os.makedirs(output, exist_ok=True)
        for k, v in self.dataframes.items():
            filename = os.path.join(output, 'mean+%s.csv' % k)
            v.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
