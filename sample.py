import dataclasses
import logging

import numpy as np

class Level:
    def __init__(self, label: int, name: str = "Classe"):
        self.label = label
        self.name = name
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


@dataclasses.dataclass
class Sample:
    filename: str
    # features: Features
    level: Level

    def __post_init__(self):
        # if self.features is None:
        #     raise ValueError("features is empty")

        if self.level is None:
            raise ValueError("level is empty")

