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
    level: Level

    def __post_init__(self):

        if self.level is None:
            raise ValueError("level is empty")


def get_level_by_name(levels, specific_epithet):
    for l in levels:
        if l.specific_epithet==specific_epithet:
            return l
    return None