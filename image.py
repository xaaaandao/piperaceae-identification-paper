import dataclasses
import os
import pathlib
from typing import LiteralString

import pandas as pd


@dataclasses.dataclass
class Image:
    color: str = dataclasses.field(default=None)
    contrast: float = dataclasses.field(default=None)
    height: int = dataclasses.field(default=None)
    width: int = dataclasses.field(default=None)
    patch: int = dataclasses.field(default=1)

    def __init__(self, data):
        self.color = data['color'].values[0]
        self.contrast = data['contrast'].values[0]
        self.height = data['height'].values[0]
        self.patch = int(data['patch'].values[0])
        self.width = data['width'].values[0]

    def save(self, output:pathlib.Path | LiteralString | str) -> None:
        """
        Salva em um arquivo CSV os valores presentes nos atributos da classe imagem.
        :param output: local onde será salvo o arquivo.
        """
        filename = os.path.join(output, 'image.csv')
        data = {'color': [], 'contrast': [], 'height': [], 'width': [], 'patch': []}
        for k, v in self.__dict__.items():
            data[k].append(v)
        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(filename, index=False, header=True, sep=';', quoting=2, encoding='utf-8')
