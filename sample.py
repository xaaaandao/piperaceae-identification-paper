import dataclasses

from level import Level


@dataclasses.dataclass
class Sample:
    filename: str = dataclasses.field(default=None)
    level: Level = dataclasses.field(default=None)
