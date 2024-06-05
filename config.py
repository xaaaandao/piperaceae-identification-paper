import dataclasses
import logging


class Config:
    def __init__(self):
        self.backend = 'loky' # value used in parameter gridsearch
        self.metrics = ['f1', 'accuracy']
        self.folds = 2
        self.cv_metric = 'f1_weighted'
        self.n_jobs = -1
        self.seed = 1234
        self.verbose = 42

    def _print(self):
        for k, v in self.__dict__.items():
            logging.info(f'{k} = {v}')




def load_cfg(config):
    if not config:
        logging.WARNING('file config not found')
        logging.WARNING('using config default')
        return

    logging.WARNING('load config file %s' % config)
