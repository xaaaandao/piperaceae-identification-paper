import collections
import dataclasses
import itertools
import logging
import os
import pathlib
from typing import LiteralString, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from config import Config
from image import Image



@dataclasses.dataclass
class Level:
    specific_epithet: str = dataclasses.field(default=None)
    label: int = dataclasses.field(default=None)

    def __eq__(self, label: int, specific_epithet: str) -> bool:
        return self.label.__eq__(label) and self.specific_epithet.__eq__(specific_epithet)


@dataclasses.dataclass
class Sample:
    filename: str = dataclasses.field(default=None)
    level: Level = dataclasses.field(default=None)


class Dataset:
    def __init__(self,
                 input: pathlib.Path | LiteralString | str,
                 count_features: int = 0,
                 count_samples: int = 0,
                 descriptor: str = None,
                 extractor: str = None,
                 format: str = None,
                 image: Image = None,
                 levels: list = [],
                 model: str = None,
                 minimum: int = 0,
                 name: str = None,
                 region: str = None,
                 samples: list = []):
        self.descriptor = descriptor
        self.extractor = extractor
        self.count_samples = count_samples
        self.count_features = count_features
        self.format = format
        self.image = image
        self.input = input
        self.levels = levels
        self.minimum = minimum
        self.model = model
        self.name = name
        self.region = region
        self.samples = samples


    def load(self):
        self.load_csv()
        self.load_samples()

    def _print(self):
        for k, v in self.__dict__.items():
            logging.info(f'{k} = {v}')

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['fold', 'specific_epithet']].drop_duplicates(subset=['specific_epithet', 'fold'], keep='last')

    def load_samples(self):
        filename = os.path.join(self.input, 'samples.csv')

        if not os.path.exists(filename):
            logging.warning('file of samples not found')

        df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8', low_memory=False, sep=';')
        dict_cols = {j: i for i, j in enumerate(df.columns)}
        self.samples = [Sample(row[dict_cols['filename']], Level(row[dict_cols['specific_epithet']], row[dict_cols['fold']])) for row in df.values]
        self.levels = [Level(row['specific_epithet'], row['fold']) for _, row in self.remove_duplicates(df).iterrows()]

    def load_csv(self):
        filename = os.path.join(self.input, 'dataset.csv')

        if not os.path.exists(filename):
            logging.warning('file of dataset not found')

        df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8', low_memory=False, sep=';')
        df = df.head(1)  # return first row
        for k in self.__dict__.keys():
            if k in df.columns and 'input' not in k:
                setattr(self, k, df[k].values[0])
        self.image = Image(df)

    def split_folds(self, config: Config, y: np.ndarray):
        np.random.seed(config.seed)
        x = np.random.rand(self.count_samples, self.count_features)
        y = [np.repeat(k, int(v / self.image.patch)) for k, v in dict(collections.Counter(y)).items()]
        y = np.array(list(itertools.chain(*y)))
        logging.info('StratifiedKFold x.shape: %s' % str(x.shape))
        logging.info('StratifiedKFold y.shape: %s' % str(y.shape))
        kf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
        return list(kf.split(x, y))

    def count_in_files(self) -> int:
        n_features = []
        for fname in sorted(pathlib.Path(self.input).rglob('*.npz')):
            try:
                data = np.load(fname)
            except:
                logging.critical('cannot load file npz')
                raise SystemExit

            n_features.append(data['x'].shape[1])

        if all(n for n in n_features):
            return n_features[0]

        logging.critical('number of features is incompatible')
        raise SystemExit

    def get_size_features(self) -> int:
        return self.count_features if self.count_features else self.count_in_files()

    def load_features(self):
        # TODO add quando é txt
        return self.load_npz() if 'npz' in self.format else None

    def load_npz(self):
        x = np.empty(shape=(0, self.get_size_features()), dtype=np.float64)
        y = []
        for fname in sorted(pathlib.Path(self.input).rglob('*.npz')):
            if fname.is_file():
                d = np.load(fname)
                x = np.append(x, d['x'], axis=0)
                y.append(d['y'])
        y = np.array(list(itertools.chain(*y)), dtype=np.int16)
        return x, y

    def get_output_name(self, classifier_name: str, count_features:int) -> str:
        model = 'empty'
        for k, v in self.__dict__.items():
            if k in ['model', 'descriptor', 'extractor'] and v:
                model = getattr(self, k)
        if self.region:
            return 'ft=%d+sam=%d+fmt=%s+clr=%s+contrast=%.2f+height=%d+width=%d+pt=%d+dt=%s+min=%d+mod=%s+region=%s+clf=%s' % \
                    (count_features, self.count_samples, self.format, \
                     self.image.color, self.image.contrast, self.image.height, self.image.width, self.image.patch, \
                     self.name, self.minimum, model, self.region, classifier_name)
        return 'ft=%d+sam=%d+fmt=%s+clr=%s+contrast=%.2f+height=%d+width=%d+pt=%d+dt=%s+min=%s+mod=%s+clf=%s' % \
                (count_features, self.count_samples, self.format, \
                 self.image.color, self.image.contrast, self.image.height, self.image.width, self.image.patch, \
                 self.name, str(self.minimum), model, classifier_name)

    def save(self, classifier, output: pathlib.Path | LiteralString | str):
        filename = os.path.join(output, 'dataset.csv')
        keys = ['descriptor','extractor','count_samples','count_features','format','input','minimum','model','name','region']
        data = dict()
        for k in keys:
            data.update({k: [getattr(self, k)]})
        data.update({'classifier': [classifier.best_estimator_.__class__.__name__],
                     'count_levels': [len(self.levels)]})
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')

        self.image.save(output)