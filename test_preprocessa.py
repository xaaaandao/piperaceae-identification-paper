from unittest import TestCase

import numpy

from dados import retorna_features_e_labels
from preprocessa import pca, retorna_valores_pca


class Test(TestCase):
    dados_teste_um = numpy.ones((375, 257))
    dados_teste_dois = numpy.ones((375, 59))
    x = None
    y = None

    @classmethod
    def setUp(self) -> None:
        self.x, self.y = retorna_features_e_labels(self.dados_teste_um)

    def test_pca(self):
        dados_normalizado = pca(self.x)
        self.assertTrue(len(dados_normalizado) == 2)

    def test_retorna_valores_pca(self):
        valores_pca = retorna_valores_pca(self.x.shape[0], self.x.shape[1])
        self.assertTrue(len(valores_pca) == 1)
        self.assertTrue(valores_pca[0] == 128)

    def test_nao_retorna_valores_pca(self):
        x, y = retorna_features_e_labels(self.dados_teste_dois)
        valores_pca = retorna_valores_pca(x.shape[0], x.shape[1])
        self.assertTrue(len(valores_pca) == 0)
