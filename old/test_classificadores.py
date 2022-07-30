from unittest import TestCase

from classificadores import Classificador, remove_lista_classificadores


class Test(TestCase):
    def test_remove_lista_classificadores(self):
        lista_classificadores = [Classificador('arvore de decisao', None, None, []), Classificador('floresta aleatoria', None, None, [])]
        self.assertTrue(len(remove_lista_classificadores(lista_classificadores)) == 1)

