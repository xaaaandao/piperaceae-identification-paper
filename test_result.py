from random import random, randint
from unittest import TestCase


def randomList(m, n):

    # Create an array of size m where
    # every element is initialized to 0
    arr = [0] * m

    # To make the sum of the final list as n
    for i in range(n) :

        # Increment any random element
        # from the array by 1
        arr[randint(0, n) % m] += 1

        # Print the generated list
    # printArr(arr, m)
    print(arr)


class TestResult(TestCase):
    def setUp(self):
        super().setUp()

    def test_a(self):
        print(randomList(235, 10503))
    # def second_dataset(self):
    #     self.total_test_no_patch = 2626
    #     self.total_train_no_patch = 7877
    #     self.patch = 3
    #     self.total_test = self.total_test_no_patch * self.patch
    #     self.total_train = self.total_train_no_patch * self.patch
    #
    # def first_dataset(self):
    #     self.total_test_no_patch = 2628
    #     self.total_train_no_patch = 7886
    #     self.patch = 3
    #     self.total_test = self.total_test_no_patch * self.patch
    #     self.total_train = self.total_train_no_patch * self.patch
