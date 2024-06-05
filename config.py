import dataclasses
import logging


@dataclasses.dataclass
class Config:
    backend: str = dataclasses.field(default='loky') # value used in parameter gridsearch
    folds: int = dataclasses.field(default=5)
    metric: str = dataclasses.field(default='f1_weighted')
    n_jobs: int = dataclasses.field(default=-1)
    seed: int = dataclasses.field(default=1234)
    verbose: int = dataclasses.field(default=42)

    def _print(self):
        for k, v in self.__dict__.items():
            logging.info(f'{k} = {v}')




def load_cfg(config):
    if not config:
        logging.WARNING('file config not found')
        logging.WARNING('using config default')
        return

    logging.WARNING('load config file %s' % config)
