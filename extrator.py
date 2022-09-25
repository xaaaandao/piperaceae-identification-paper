import numpy
import os
import re

import dados
import preprocessa

from classificadores import classifica
from log import console_log, escreve_log_arquivo_console


def retorna_n_patches(filename):
    _, patches = re.split('_', filename)
    _, n_patches = re.split('-', patches)
    return int(n_patches)


def retorna_n_blocos(nome_arquivo):
    return int(re.search('features_(.+?)_(.+?)x(.+?).txt', nome_arquivo).group(2))


def retorna_hog_n_blocos(nome_arquivo):
    return int(re.search('hog_(.+?)_(.+?)_(.+?).txt', nome_arquivo).group(2))


def retorna_extrator(nome_arquivo):
    lista_extratores = ['MobileNetV2', 'Resnet50V2', 'VGG16', 'HOG', 'SURF128', 'LBP', 'SIFT', 'BSIF', 'OBIF2']
    for extrator in lista_extratores:
        if extrator.lower() in str(nome_arquivo).lower():
            return extrator


def retorna_n_folds(filename):
    fold, _ = re.split('_', filename)
    _, n_fold = re.split('-', fold)
    return int(n_fold)
    # return int(re.search('fold-(.+?)_patches-(.+?).npy', nome_arquivo).group(1))


def adiciona_coluna_label(todos_dados, dados, nome_arquivo):
    for feature in dados:
        todos_dados.append(numpy.append(feature, retorna_n_folds(nome_arquivo)))


def testa(dataset, cv, indices_cv, pca=False):
    console_log.info(f'usando o {dataset}')

    extrator = None
    dados = None
    print(os.path.exists(dataset))
    if os.path.isdir(dataset):
        extrator, dados = mobilenet_resnet_vgg(cv, dataset, indices_cv, pca)
    elif os.path.isfile(dataset):
        if '.txt' in dataset:
            if 'features' in dataset:
                extrator, dados = bsif_lbp_lpq_obif_surf(cv, dataset, indices_cv, pca)
            else:
                extrator, dados = surf_sift_lbp_hog(cv, dataset, indices_cv, pca)
                console_log.info('a')
        else:
            escreve_log_arquivo_console(f'dataset {dataset} invalido')
    carrega_dados(dados, extrator, cv, pca)
    classifica(extrator, indices_cv)


def bsif_lbp_lpq_obif_surf(cv, dataset, indices_cv, pca):
    n_blocos = None
    if 'features' in dataset:
        n_blocos = retorna_n_blocos(dataset)
        n_blocos = n_blocos * n_blocos
    extrator = Extrator(retorna_extrator(dataset), n_blocos)
    # print(f'e: {extrator.nome}, n_blocos: {extrator.n_patches}')
    carrega_dados(numpy.loadtxt(dataset), extrator, cv, pca)
    # classifica(extrator, indices_cv)
    return extrator, numpy.loadtxt(dataset)


def surf_sift_lbp_hog(cv, dataset, indices_cv, pca):
    extrator = Extrator(retorna_extrator(dataset), None)
    # carrega_dados(numpy.loadtxt(dataset), extrator, cv, pca)
    # classifica(extrator, indices_cv)
    return extrator, numpy.loadtxt(dataset)


def retorna_divisao(dataset):
    lista_divisao = ['Horizontal', 'Bloco', 'Vertical']
    for d in lista_divisao:
        if 'Bloco' in d.lower():
            return '+Block'
        if d.lower() in dataset:
            return f'+{d}'
    return ''


def mobilenet_resnet_vgg(cv, dataset, indices_cv, pca):
    todos_dados = []
    n_patch = None
    for nome_arquivo in sorted(os.listdir(dataset)):
        if '.npy' in nome_arquivo:
            dados = numpy.load(os.path.join(dataset, nome_arquivo))
            adiciona_coluna_label(todos_dados, dados, nome_arquivo)
            n_patch = retorna_n_patches(nome_arquivo)
    if 'bloco' in dataset and n_patch > 1:
        n_patch = n_patch * n_patch
    extrator = Extrator(f'{retorna_extrator(dataset)}{retorna_divisao(dataset)}', n_patch)
    carrega_dados(numpy.array(todos_dados), extrator, cv, pca)
    # classifica(extrator, indices_cv)
    return extrator, numpy.array(todos_dados)


def carrega_dados(dataset, extrator, cv, pca):
    # print(numpy.isnan(dataset))
    # dataset = numpy.loadtxt(dataset)
    if numpy.isnan(dataset).any():
        escreve_log_arquivo_console(f'arquivo {dataset} contem NaN')

    x, y = dados.retorna_features_e_labels(dataset)
    lista_dados = preprocessa.preprocessa(x, pca)
    define_labels(lista_dados, y)
    define_cv(cv, lista_dados)
    extrator.lista_dados = lista_dados


def define_labels(lista_dados, y):
    [d.__setattr__('y', y) for d in lista_dados]


def define_cv(cv, lista_dados):
    [d.__setattr__('cv', cv) for d in lista_dados]


class Extrator:

    def __init__(self, nome, n_patches) -> None:
        super().__init__()
        self.nome = nome
        self.n_patches = n_patches
        self.lista_dados = None
