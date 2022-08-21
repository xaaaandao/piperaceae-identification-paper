from unittest import TestCase

import numpy

from old.result import get_index_max_value, next_sequence, y_test_with_patch, prod_all_prob, sum_all_prob, \
    convert_prob_to_label, Result, sum_all_results, prod_all_results, max_all_results


class TestResult(TestCase):
    array_ond_one = numpy.ones(shape=(7,))
    array_oned_two = numpy.array([1, 3, 5, 7, 9])
    array_oned_three = numpy.array([2, 4, 6, 8, 10])
    array_twod_one = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    array_twod_two = numpy.array([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
    cfg = {"n_labels": 5}
    list_result_sum = list([])
    list_result_prod = list([])
    list_result_max = list([])

    def test_get_index_max_value(self):
        self.assertEqual(4, get_index_max_value(self.array_twod_one))

    def test_next_sequence(self):
        self.assertEqual(1, len(list(next_sequence(0, self.array_twod_one.shape[0], 2))))

    def test_y_test_with_patch(self):
        self.assertTrue(y_test_with_patch(7, self.array_ond_one))

    def test_prod_all_prob(self):
        y_pred_prob, y_pred = prod_all_prob(self.cfg, 2, self.array_twod_one)
        result_pred_prob = numpy.array([[0, 6, 14, 24, 36]])
        result_pred = numpy.array([5])
        self.assertTrue((result_pred_prob == y_pred_prob).all())
        self.assertTrue((result_pred == y_pred).all())

    def test_sum_all_prob(self):
        y_pred_prob, y_pred = sum_all_prob(self.cfg, 2, self.array_twod_one)
        result_pred_prob = numpy.array([[5, 7, 9, 11, 13]])
        result_pred = numpy.array([5])
        self.assertTrue((result_pred_prob == y_pred_prob).all())
        self.assertTrue((result_pred == y_pred).all())

    def test_convert_prob_to_label(self):
        self.assertTrue((convert_prob_to_label(self.array_twod_one) == numpy.array([5, 5])).all())

    def test_sum_all_results(self):
        self.create_list_result_sum()
        result = sum_all_results(self.list_result_sum)
        self.assertTrue((result == numpy.array([[10, 12, 14, 16, 18], [20, 22, 24, 26, 28]])).all())

    def test_prod_all_results(self):
        self.create_list_result_prod()
        result = prod_all_results(self.list_result_prod)
        self.assertTrue((result == numpy.array([[0, 11, 24, 39, 56], [75, 96, 119, 144, 171]])).all())

    def test_max_all_results(self):
        self.create_list_result_max()
        result = max_all_results(self.list_result_max)
        self.assertTrue((result == self.array_oned_three).all())

    def create_list_result_sum(self):
        self.list_result_sum.append(Result(None, None, "sum", self.array_twod_one, numpy.zeros(shape=(1, )), numpy.zeros(shape=(1, ))))
        self.list_result_sum.append(Result(None, None, "sum", self.array_twod_two, numpy.zeros(shape=(1, )), numpy.zeros(shape=(1, ))))

    def create_list_result_prod(self):
        self.list_result_prod.append(Result(None, None, "prod", self.array_twod_one, numpy.zeros(shape=(1, )), numpy.zeros(shape=(1, ))))
        self.list_result_prod.append(Result(None, None, "prod", self.array_twod_two, numpy.zeros(shape=(1, )), numpy.zeros(shape=(1, ))))

    def create_list_result_max(self):
        self.list_result_max.append(Result(None, None, "max", numpy.zeros(shape=(1, )), self.array_oned_two, self.array_oned_two))
        self.list_result_max.append(Result(None, None, "max", numpy.zeros(shape=(1, )), self.array_oned_three, self.array_oned_three))