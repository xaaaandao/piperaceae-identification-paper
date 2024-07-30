import csv
import itertools
import logging

import numpy as np
import pandas as pd

import fold
from dataset import Dataset
from evaluate import TopK


class Mean:
    def __init__(self, folds: list):
        self.dfs = [f.dataframes for f in folds]
        self.c = self.means()
        self.best()
        self.mean_top = self.top()
        self.true_positive()

    def means(self):
        evals = pd.concat(df['evaluations'] for df in self.dfs)
        a = evals.groupby('rule').mean().reset_index()
        b = evals.groupby('rule').std().reset_index()
        a.rename(columns={'accuracy': 'mean_accuracy', 'f1': 'mean_f1'}, inplace=True)
        b.rename(columns={'accuracy': 'std_accuracy', 'f1': 'std_f1'}, inplace=True)
        return a.merge(b, how='inner', on='rule')

    def best(self):
        df_accuracy = self.c.loc[self.c['mean_accuracy'].idxmax()]
        df_f1 = self.c.loc[self.c['mean_f1'].idxmax()]
        data = {'metric': ['mean_accuracy', 'mean_f1', 'std_accuracy', 'std_f1'],
                'value': [df_accuracy['mean_accuracy'], df_f1['mean_f1'], df_accuracy['std_accuracy'], df_f1['std_f1']],
                'rule': [df_accuracy['rule'], df_f1['rule'], df_accuracy['rule'], df_f1['rule']]}
        a= pd.DataFrame(data, columns=list(data.keys()))
        print(a)

    def top(self):
        evals = pd.concat(df['topk'] for df in self.dfs)
        a = evals.groupby(['k', 'rule'])['topk_accuracy_score'].mean().reset_index()
        b = evals.groupby(['k', 'rule'])['topk_accuracy_score'].std().reset_index()
        c = evals.groupby(['k', 'rule'])['count_test'].mean().reset_index()
        d = evals.groupby(['k', 'rule'])['count_test'].std().reset_index()

        a.rename(columns={'topk_accuracy_score': 'mean_topk_accuracy_score'}, inplace=True)
        b.rename(columns={'topk_accuracy_score': 'std_topk_accuracy_score'}, inplace=True)
        c.rename(columns={'count_test': 'mean_count_test'}, inplace=True)
        d.rename(columns={'count_test': 'std_count_test'}, inplace=True)
        e = a.merge(b, how='inner', on=['rule', 'k'])
        f = c.merge(d, how='inner', on=['rule', 'k'])
        g = e.merge(f, how='inner', on=['rule', 'k'])
        print(g)
        g['mean_topk_accuracy_score+100'] = (g['mean_topk_accuracy_score']) / g['mean_count_test']
        g.to_csv('a.csv', sep=';', quoting=2)
        return g
        # break

    def true_positive(self):
        if any('true_positive' not in df for df in self.dfs):
            return None

        evals = pd.concat(df['true_positive'] for df in self.dfs)
        a = evals.groupby(['rule', 'labels'])['true_positive'].mean().reset_index()
        b = evals.groupby(['rule', 'labels'])['true_positive'].std().reset_index()
        c = evals.groupby(['rule', 'labels'])['count_train'].mean().reset_index()
        d = evals.groupby(['rule', 'labels'])['count_train'].std().reset_index()
        e = evals.groupby(['rule', 'labels'])['count_test'].mean().reset_index()
        f = evals.groupby(['rule', 'labels'])['count_test'].std().reset_index()
        a.rename(columns={'true_positive': 'mean_true_positive'}, inplace=True)
        b.rename(columns={'true_positive': 'std_true_positive'}, inplace=True)
        c.rename(columns={'count_train': 'mean_count_train'}, inplace=True)
        d.rename(columns={'count_train': 'std_count_train'}, inplace=True)
        e.rename(columns={'count_test': 'mean_count_test'}, inplace=True)
        f.rename(columns={'count_test': 'std_count_test'}, inplace=True)
        g = a.merge(b, how='inner', on=['rule', 'labels'])
        h = c.merge(d, how='inner', on=['rule', 'labels'])
        i = e.merge(f, how='inner', on=['rule', 'labels'])
        j = g.merge(h, how='inner', on=['rule', 'labels'])
        k = i.merge(j, how='inner', on=['rule', 'labels'])
        print(k)
        k.to_csv('d.csv', sep=';', quoting=2)
        # break