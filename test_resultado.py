from unittest import TestCase

import numpy

from resultado import regra_da_soma, regra_do_produto, regra_do_maior, todos_valores_label_iguais


class Test(TestCase):
    dataset_um = numpy.array([numpy.arange(1, 6), numpy.arange(1, 6)])
    dataset_dois = numpy.array([numpy.array([1, 2, 3, 4, 5]), numpy.array([1, 2, 6, 8, 4])])
    dataset_tres = numpy.array([numpy.array([1, 2, 3, 4, 8]), numpy.array([1, 2, 3, 8, 5])])
    dataset_quatro = numpy.zeros((375, 512))
    dataset_cinco = numpy.ones((375, 512))
    y_pred = numpy.empty((0,))

    def test_regra_da_soma(self):
        y_pred = regra_da_soma(self.dataset_um, self.y_pred)
        self.assertEqual(y_pred[0], 5)

    def test_regra_do_produto(self):
        y_pred = regra_do_produto(self.dataset_um, self.y_pred)
        self.assertEqual(y_pred[0], 5)

    def test_regra_do_maior_valor_tem_repetido_label_iguais(self):
        # quando os dois maiores valores estao na posicao cinco
        self.y_pred = regra_do_maior(self.y_pred, self.dataset_um, [])
        self.assertEqual(self.y_pred[0], 5)

    def test_regra_do_maior_valor_tem_somente_um(self):
        self.y_pred = regra_do_maior(self.y_pred, self.dataset_dois, [])
        self.assertEqual(self.y_pred[0], 4)

    def test_regra_do_maior_valor_tem_repetido_label_diferentes(self):
        self.y_pred = regra_do_maior(self.y_pred, self.dataset_tres, [])
        self.assertEqual(self.y_pred[0], 5)

    def test_todos_valores_label_iguais(self):
        todos_valores_label_iguais
        self.fail()

