from log import console_log

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import dados as d


def retorna_valores_pca(n_amostras, n_features):
    lista_valor_pca = []
    if n_features > 128:
        i = 7
        while pow(2, i) < n_amostras and pow(2, i) < n_features:
            if n_features < (pow(2, i) - 5) or n_features > (pow(2, i) + 5): # para que nao seja valores proximos do original...
                lista_valor_pca.append(pow(2, i))
            i += 1
        return lista_valor_pca
    return lista_valor_pca


def pca(x):
    lista_dados_normalizados_pca = []
    lista_valores_pca = retorna_valores_pca(x.shape[0], x.shape[1])
    console_log.info(f'valores de pca {lista_valores_pca}')
    for valor in lista_valores_pca:
        p = PCA(n_components=valor)
        novo_x = p.fit_transform(x)
        lista_dados_normalizados_pca.append(d.Dados(novo_x, None, novo_x.shape[0], novo_x.shape[1], valor))
    lista_dados_normalizados_pca.append(d.Dados(x, None, x.shape[0], x.shape[1], x.shape[1]))
    return lista_dados_normalizados_pca


def normaliza(x):
    normalizador = StandardScaler()
    return normalizador.fit_transform(x)


def preprocessa(x, aplica_pca):
    console_log.info(f'normalizando...')
    x_normalizado = normaliza(x)
    if aplica_pca:
        return pca(x_normalizado)
    return [d.Dados(x_normalizado, None, x_normalizado.shape[0], x_normalizado.shape[1], x_normalizado.shape[1])]