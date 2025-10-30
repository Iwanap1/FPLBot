from dataclasses import dataclass
from auto_fpl.team import Squad
from typing import List, Tuple

@dataclass
class TransferState:
    current_squad: Squad
    transfers_made: List[Tuple[int, int]]  # list of (out_player_id, in_player_id)
    accumilated_xp_gain: float             # including transfer hits
