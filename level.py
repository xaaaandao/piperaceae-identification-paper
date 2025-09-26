import dataclasses

import numpy as np


@dataclasses.dataclass
class Level:
    # label: int = dataclasses.field(default=None)
    # specific_epithet: str = dataclasses.field(default=None)

    def __init__(self, label, specific_epithet):
        self.label = label
        self.specific_epithet = specific_epithet.replace("+", " ")


    def __eq__(self, label: int, specific_epithet: str) -> bool:
        """
        Verifica se o inteiro do label (f1, f2, f3, ...) é igual ao valor que está na classe

        :param label: rótulo que identifica a classe.
        :param specific_epithet: nome da espécie.
        :return: bool, True se ambas as informações forem iguais.
        """
        return self.label.__eq__(label) and self.specific_epithet.__eq__(specific_epithet)

class LevelTP(Level):
    def __init__(self, label, specific_epithet, tp):
        super().__init__(label, specific_epithet)
        self.true_positive = tp

def get_level_by_name(levels, specific_epithet):
    for l in levels:
        if l.specific_epithet==specific_epithet.replace("+", " "):
            return l
    return None

def get_level_by_label(label, levels):
    for l in levels:
        if l.label==label:
            return l
    return None
