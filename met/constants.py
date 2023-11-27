import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class Constants:
    HERE = pathlib.Path(__file__)
    SRC = HERE.parents[0]
    REPO = HERE.parents[1]
    DATA = REPO.joinpath("data")
    OUTPUTS = REPO.joinpath("outputs")
    SEED = 43
