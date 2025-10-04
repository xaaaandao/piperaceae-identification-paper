import numpy as np
from sklearn.decomposition import PCA

from config import Config
from dataset import Dataset


def has_pca(config: Config, dataset: Dataset, extractors: dict, x: np.ndarray) -> list:
    """
    Aplica PCA no conjunto de dados. Os valores estão definidos em um dicionário.
    :param config: classe config com os valores das configurações dos experimentos.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param extractors: dicionário com os extratores
    :param x: matriz com as features.
    :return: list, lista com as features reduzidas.
    """
    return [PCA(n_components=d, random_state=config.seed).fit_transform(x) for d in extractors[dataset.model.lower()] if d < dataset.features]


def apply_pca(config: Config, dataset: Dataset, extractors: dict, pca: bool, x: np.ndarray) -> list:
    """
    Verifica se é necessário aplicar o PCA.
    :param config: classe config com os valores das configurações dos experimentos.
    :param dataset: classe dataset com informações do conjunto de dados.
    :param extractors: dicionário com os extratores
    :param pca: booleano que indica se é necessário aplicar ou não o PCA.
    :param x: matriz com as features.
    :return: list, lista com as features reduzidas.
    """
    return has_pca(config, dataset, extractors, x) if pca and dataset.model else [x]
