import itertools
import logging

import numpy as np
import pandas as pd

import fold
from evaluate import TopK


class Mean:
    def __init__(self, folds: list):
        self.dfs = [f.dfs for f in folds]
        self.means()

    def means(self):
        evals = pd.concat(df['evaluations'] for df in self.dfs)
        a = evals.groupby('rule').mean().reset_index()
        b = evals.groupby('rule').std().reset_index()
        a.rename(columns={'accuracy': 'mean_accuracy', 'f1': 'mean_f1'}, inplace=True)
        b.rename(columns={'accuracy': 'std_accuracy', 'f1': 'std_f1'}, inplace=True)
        return a.merge(b, how='inner', on='rule')




