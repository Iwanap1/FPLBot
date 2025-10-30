from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Player:
    pid: int
    team_id: int
    price: float           # in millions, i.e. price from the API divided by 10
    pos: str               # {"GK","DEF","MID","FWD"}
    xps: List[float]

