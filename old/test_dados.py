from unittest import TestCase

import numpy

from dados import retorna_features_e_labels, retorna_dados_imgs_patch


class Test(TestCase):
    dados_teste = numpy.zeros((1125, 512))

    def test_retorna_dados_imgs_patch(self):
        x, y = retorna_features_e_labels(self.dados_teste)
        posicoes = tuple(numpy.random.choice(numpy.arange(1, 375), 300, replace=False))
        novo_x, novo_y = retorna_dados_imgs_patch(x, y, posicoes, 3)
        self.assertTrue(novo_x.shape[0] == 900)
        self.assertTrue(novo_x.shape[1] == 511)
