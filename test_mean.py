import string
import random
from unittest import TestCase

import numpy as np

from config import Config
from dataset import Dataset
from evaluate import Evaluate
from image import Image
from level import Level
from predict import Predict
from result import Result


class TestMean(TestCase):

    # def create_evals(self):
        # self.evals = [Evaluate(self.levels, self.y_pred, self.y_pred_proba, self.y_true)]

    def create_levels(self):
        return [Level('a', 1),
                       Level('b', 2),
                       Level('c', 3),
                       Level('d', 4),
                       Level('e', 5),
                       Level('f', 6)]

    def setUp(self):
        super().setUp()
        self.levels = self.create_levels()



