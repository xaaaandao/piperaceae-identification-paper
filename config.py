import dataclasses
import logging
import os
import pathlib
from typing import LiteralString

import pandas as pd


class Config:
    def __init__(self):
        self.backend = 'loky' # value used in parameter gridsearch
        self.metrics = ['f1', 'accuracy']
        self.folds = 5
        self.cv_metric = 'f1_weighted'
        self.n_jobs = -1
        self.seed = 1234
        self.verbose = 42

    def _print(self):
        for k, v in self.__dict__.items():
            logging.info(f'{k} = {v}')

    def save(self, output: pathlib.Path | LiteralString | str) -> None:
        filename = os.path.join(output, 'config.csv')
        data = dict()
        for k, v in self.__dict__.items():
            data.update({k: [v]})
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')


def load_cfg(config):
    if not config:
        logging.WARNING('file config not found')
        logging.WARNING('using config default')
        return

    logging.WARNING('load config file %s' % config)
