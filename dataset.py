import collections
import itertools
import logging
import os
import pathlib
from typing import LiteralString
from unittest import case

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from config import Config
from image import Image
from level import Level
from sample import Sample


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
        self.load_dataset()
        self.load_samples()

    def _print(self):
        for k, v in self.__dict__.items():
            logging.info(f'{k} = {v}')

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['fold', 'specific_epithet']].drop_duplicates(subset=['specific_epithet', 'fold'], keep='last')

    def load_samples(self):
        """
        Carrega as informações o arquivo CSV que contém as informações das amostras.
        """
        filename = os.path.join(self.input, 'samples.csv')

        if not os.path.exists(filename):
            logging.warning('file of samples not found')

        df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8', low_memory=False, sep=';')
        self.set_samples(df)
        self.set_levels(df)

    def set_samples(self, df:pd.DataFrame):
        """
        O atributo samples recebe todas as informações das amostras utilizadas.
        :param df: DataFrame com as informações das amostras.
        """
        dict_cols = {j: i for i, j in enumerate(df.columns)}
        self.samples = [Sample(row[dict_cols['filename']],
                               Level(row[dict_cols['specific_epithet']],
                                     row[dict_cols['fold']])) for row in df.values]

    def set_levels(self, df:pd.DataFrame):
        """
        O atributo levels recebe todas os levels (espécies) que estão sendo utilizados.
        Ele utiliza o arquivo de amostras para capturar todas as espécies disponíveis.
        :param df: DataFrame com as informações das amostras.
        """
        df = self.remove_duplicates(df)
        dict_cols = {j: i for i, j in enumerate(df.columns)}
        self.levels = [Level(row[dict_cols['specific_epithet']],
                             row[dict_cols['fold']])
                       for row in df.values]

    def load_dataset(self):
        """
        Carrega as informações o arquivo CSV que contém as informações do dataset.
        """
        filename = os.path.join(self.input, 'dataset.csv')

        if not os.path.exists(filename):
            logging.warning('file of dataset not found')

        df = pd.read_csv(filename, index_col=None, header=0, encoding='utf-8', low_memory=False, sep=';')
        df = df.head(1)  # return first row
        self.set_dataset(df)
        self.image = Image(df)

    def set_dataset(self, df):
        """
        Os atributos dessa classe recebem os valores presentes dentro do arquivo CSV.
        :param df: DataFrame com as informações do dataset.
        """
        for k in self.__dict__.keys():
            if 'region' in df['name'].values[0]:
                setattr(self, 'region', df['regions'].values[0])
            if k in df.columns and 'input' not in k:
                setattr(self, k, df[k].values[0])


    def split_folds(self, config: Config, y: np.ndarray):
        """
        Essa função gera os indíces de treino e de teste de cada fold, porém eles só são gerados se for passado as features e as classes que essas features pertencem. Essa função considera a quantidade features originais.
        Por exemplo, se o conjunto de dados é formado por 45 imagens (15 imagens divididas em três partes), é considerado 15.
        :param config:
        :param y: vetor de uma dimensão que contém os rótulos que aquela classe pertence.
        :return: None.
        """
        np.random.seed(config.seed)
        x = np.random.rand(self.count_samples, self.count_features)
        y = [np.repeat(k, int(v / self.image.patch)) for k, v in dict(collections.Counter(y)).items()]
        y = np.array(list(itertools.chain(*y)))
        logging.info('StratifiedKFold x.shape: %s' % str(x.shape))
        logging.info('StratifiedKFold y.shape: %s' % str(y.shape))
        kf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
        return list(kf.split(x, y))

    def count_in_files(self) -> int:
        """
        Concatena todos os arquivos npz para contabilizar a quantidade de amostras.
        :return: int, com a quantidade de amostras que aquele diretório possui.
        """
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
        """
        Verifica se o atributo count_features tem um valor.
        Caso positivo, retorna o valor presente no atributo.
        Caso contrário, contabiliza por meio dos arquivos presentes no diretório.
        :return:int, com a quantidade de amostras presentes no dataset.
        """
        return self.count_features if self.count_features else self.count_in_files()

    def load_features(self):
        # TODO add quando é txt
        # return self.load_npz() if 'npz' in self.format else self.
        match self.format:
            case 'npz':
                return self.load_npz()
            case 'csv':
                return self.load_csv()

    def load_npz(self):
        """
        Carrega em uma única matriz todas as features de todas as classes.
        Por fim, é gerado um numpy array com as classes que pertencem aquelas features.
        :return: x, numpy array com todas as features.
        :return: y, numpy array com as classes que pertencem aquelas features.
        """
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
        """
        Cria o nome da pasta (baseado nos atributos presentes) aonde ficarão salvos os resultados.
        :param classifier_name: nome do classificador.
        :param count_features: números de features.
        :return: str, nome da pasta.
        """
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
        """
        Salva em um arquivo CSV todos os valores presentes no atributos da classe
        Dataset.
        :param classifier: nome do classificador utilizado.
        :param output: local aonde o arquivo CSV será salvo.
        """
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

    def load_csv(self):
        dfs = []
        fold = 0
        y = []

        for p in sorted(pathlib.Path(self.input).rglob('*.csv')):
            if 'dataset' not in p.stem and 'samples' not in p.stem:
                fold = p.stem
                fold = int(fold.replace('f', ''))
                df = pd.read_csv(p, sep=',', escapechar='\n')
                d = df.iloc[0:, 1:]
                dfs.append(d)
                y.append([fold] * d.shape[0])


        concatenated_df = pd.concat(dfs, axis=0)
        X = concatenated_df.to_numpy()
        Y = np.array(list(itertools.chain(*y))).astype(np.int16)
        return X, Y



