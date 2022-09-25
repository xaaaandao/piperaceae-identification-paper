import numpy


def retorna_dados_imgs_patch(x, y, posicao, n_patches):
    novo_x = numpy.zeros(shape=(0, x.shape[1]))
    novo_y = numpy.zeros(shape=(0,))

    for p in posicao:
        comeca = (p * n_patches)
        termina = comeca + n_patches
        novo_x = numpy.concatenate([novo_x, x[comeca:termina]])
        novo_y = numpy.concatenate([novo_y, y[comeca:termina]])

    return novo_x, novo_y


def retorna_features_e_labels(dados):
    n_amostras, n_features = dados.shape
    return dados[0:, 0:n_features - 1], dados[:, n_features - 1]


class Dados:
    def __init__(self, x, y, n_amostras, n_features, valor_pca, cv=None) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.n_amostras = n_amostras
        self.n_features = n_features
        self.labels = numpy.unique(y)
        self.valor_pca = valor_pca
        self.cv = cv
        self.lista_classificadores = None

