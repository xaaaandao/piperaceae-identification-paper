import dataclasses


@dataclasses.dataclass
class Level:
    specific_epithet: str = dataclasses.field(default=None)
    label: int = dataclasses.field(default=None)

    def __init__(self, label, specific_epithet):
        super().__init__()
        self.label = label
        self.specific_epithet = str(specific_epithet)

    def __eq__(self, label: int, specific_epithet: str) -> bool:
        """
        Verifica se o inteiro do label (f1, f2, f3, ...) é igual ao valor que está na classe

        :param label: rótulo que identifica a classe.
        :param specific_epithet: nome da espécie.
        :return: bool, True se ambas as informações forem iguais.
        """
        return self.label.__eq__(label) and self.specific_epithet.__eq__(specific_epithet)
