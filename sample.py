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

# class Features:
#     def __init__(self, data):
#         self.data = data
#         self.x = np.array([])
#         self.y = np.array([])
#         self.split()
#         self.set_type()
#
#     def split(self):
#         self.x = self.data[:, :-2]
#         self.y = self.data[:, -2]
#
#     def set_type(self):
#         self.x = self.x.astype(float)
#         self.y = self.y.astype(float).astype(np.int16)
#         logging.info("x.shape: %s" % str(self.x.shape))
#         logging.info("y.shape: %s" % str(self.y.shape))


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

