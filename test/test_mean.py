from unittest import TestCase

import numpy as np
import pandas as pd

from dataset import Dataset
from fold import Fold
from level import Level
from mean import Mean


class TestMean(TestCase):
    n_levels = 23

    def create_dataset(self):
        self.dataset = Dataset('./test/files')
