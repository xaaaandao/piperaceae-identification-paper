import itertools
import logging
import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd


class Result:
    def __init__(self, count_train: dict, count_test: dict, patch: int, predicts: list, time: float):
        self.count_test = count_test
        self.count_train = count_train
        self.predicts = predicts
        self.time = time
        self.total_test = np.sum([v for v in count_test.values()])
        self.total_train = np.sum([v for v in count_train.values()])
        self.total_test_no_patch = np.sum([int(v / patch) for v in count_test.values()])
        self.total_train_no_patch = np.sum([int(v / patch) for v in count_train.values()])

    def save(self, levels:list, output: pathlib.Path | LiteralString | str, patch:int):
        self.save_info_result(output)
        self.save_predicts(levels, output)
        self.save_evaluations(levels, output, patch)

    def save_evaluations(self, levels:list, output: pathlib.Path | LiteralString | str, patch:int):
        data = {'f1': [], 'accuracy': [], 'rule': []}
        df = pd.DataFrame(data, columns=data.keys())

        data = {'k': [], 'topk_accuracy_score': [], 'rule': []}
        df_topk = pd.DataFrame(data, columns=data.keys())

        for predict in self.predicts:
            df = pd.concat([df, predict.save()], axis=0)
            predict.eval.save(self.count_train, self.count_test, levels, output, patch, predict.rule)
            df_topk = pd.concat([df_topk, predict.eval.save_topk(self.total_test_no_patch, levels, output, predict.rule)], axis=0)

        filename = os.path.join(output, 'evaluations.csv')
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

        filename = os.path.join(output, 'topk.csv')
        df_topk.sort_values(['k', 'rule'], ascending=[True, False], inplace=True)
        df_topk.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
        self.save_best(df, output)

    def save_best(self, df: pd.DataFrame, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'best+evals.csv')
        data = {'value': [df.query('f1 == f1.max()')['f1'].values[0],
                          df.query('accuracy == accuracy.max()')['accuracy'].values[0]],
                'metric': ['f1', 'accuracy'],
                'rule': [df.query('f1 == f1.max()')['rule'].values[0],
                         df.query('accuracy == accuracy.max()')['rule'].values[0]]}
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_info_result(self, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'info_results.csv')

        data = {
            'time': [self.time],
            'total_test': [self.total_test],
            'total_train': [self.total_train],
            'total_test_no_patch': [self.total_test_no_patch],
            'total_train_no_patch': [self.total_train_no_patch]
        }
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

    def save_predicts(self, levels: list, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'predicts.csv')

        data = {
            'y_pred+sum': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('sum')])),
            'y_pred+mult': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('mult')])),
            'y_pred+max': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('max')])),
            'y_true': list(itertools.chain(*[self.predicts[0].y_true.tolist()]))
        }
        df = pd.DataFrame(data, columns=data.keys())

        if len(levels) > 0:
            df = df.applymap(lambda row: list(filter(lambda x: x.label.__eq__(row), levels))[0].specific_epithet)

        df['equals'] = df.apply(lambda row: row[row == row['y_true']].index.tolist(), axis=1)
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
        logging.info('Saving %s' % filename)





