import itertools
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

    def save(self, levels:list, output: pathlib.Path | LiteralString | str):
        self.save_info_result(output)
        self.save_predicts(output)
        self.save_evaluations(levels, output)

    def save_evaluations(self, levels:list, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'evaluations.csv')

        data = {'f1': [], 'accuracy': [], 'rule': []}
        df = pd.DataFrame(data, columns=data.keys())
        for predict in self.predicts:
            print(predict.eval, type(predict.eval))
            df = pd.concat([df, predict.save()], axis=0)
            predict.eval.save_confusions_matrixs(self.count_train, self.count_test, levels, output, predict.rule)
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

    def save_predicts(self, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'predicts.csv')

        data = {
            'y_pred+sum': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('sum')])),
            'y_pred+mult': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('mult')])),
            'y_pred+max': list(itertools.chain(*[p.y_pred.tolist() for p in self.predicts if p.rule.__eq__('max')])),
            'y_true': list(itertools.chain(*[self.predicts[0].y_true.tolist()]))
        }
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
