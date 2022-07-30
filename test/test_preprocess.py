import numpy
import unittest

from preprocess import get_all_values_pca


class TestPreprocess(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_should_not_values_pca(self):
        values_pca = get_all_values_pca(numpy.zeros(shape=(375, 59)))
        self.assertEqual(len(values_pca), 0)

    def test_should_not_exceed_n_samples(self):
        values_pca = get_all_values_pca(numpy.zeros(shape=(375, 513)))
        self.assertEqual(len(values_pca), 2)
