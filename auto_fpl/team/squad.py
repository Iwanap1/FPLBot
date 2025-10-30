from typing import Dict, List, Tuple
from .player import Player
import numpy as np

POS_ORDER = ["GK", "DEF", "MID", "FWD"]
SQUAD_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3



class Squad:
    def __init__(self, players: List[Player], bank: float, free_transfers: int = 1):
        assert len(players) == 15, "Squad must have 15 players"
        self.players: List[Player] = list(players)
        self.bank = float(bank)
        self.free_transfers = int(np.clip(free_transfers, 1, 2))
        self.all_pids = set(p.pid for p in players)

    def club_counts(self) -> Dict[int, int]:
        cc: Dict[int, int] = {}
        for p in self.players:
            cc[p.team_id] = cc.get(p.team_id, 0) + 1
        return cc

    def pos_counts(self) -> Dict[str, int]:
        pc = {k: 0 for k in SQUAD_REQUIREMENTS}
        for p in self.players:
            pc[p.pos] += 1
        return pc

    def can_swap(self, out_idx: int, incoming: Player) -> bool:
        if out_idx < 0 or out_idx >= 15:
            return False
        out_p = self.players[out_idx]
        # must match position
        if incoming.pos != out_p.pos:
            return False
        # NEW: forbid duplicates (except replacing same pid)
        if incoming.pid != out_p.pid and any(
            p.pid == incoming.pid for i, p in enumerate(self.players) if i != out_idx
        ):
            return False

        # club limit
        cc = self.club_counts()
        cc[out_p.team_id] -= 1  # freeing one slot from the outgoing player's club
        if cc.get(incoming.team_id, 0) + 1 > MAX_PER_CLUB:
            return False
        cost_delta = incoming.price - out_p.price
        return self.bank - cost_delta >= -1e-9


    def apply_swap(self, out_idx: int, incoming: Player) -> bool:
        if not self.can_swap(out_idx, incoming):
            return False
        out_p = self.players[out_idx]
        self.players[out_idx] = incoming
        self.bank -= (incoming.price - out_p.price)
        self.all_pids = set(p.pid for p in self.players)
        return True

    def best_xi(self, gw_onlook: int) -> Tuple[List[Player], float]:
        def xp(p: Player, gw: int) -> float:
            try:
                return float(p.xps[gw])
            except Exception:
                return 0.0

        # Split squad by position and sort each by GW xP (desc)
        gks  = [p for p in self.players if p.pos == "GK"]
        defs = [p for p in self.players if p.pos == "DEF"]
        mids = [p for p in self.players if p.pos == "MID"]
        fwds = [p for p in self.players if p.pos == "FWD"]

        gks.sort(key=lambda p: xp(p, gw_onlook), reverse=True)
        defs.sort(key=lambda p: xp(p, gw_onlook), reverse=True)
        mids.sort(key=lambda p: xp(p, gw_onlook), reverse=True)
        fwds.sort(key=lambda p: xp(p, gw_onlook), reverse=True)

        # Valid outfield formations: DEF 3–5, MID 2–5, FWD 1–3 with total outfield = 10
        valid_formations = [
            (d, m, f)
            for d in range(3, 6)
            for m in range(2, 6)
            for f in range(1, 4)
            if d + m + f == 10
        ]
        best_total_xp = float("-inf")
        best_eleven: List[Player] = []
        if not gks:
            return [], 0.0
        best_gk = gks[0]
        for d, m, f in valid_formations:
            if len(defs) < d or len(mids) < m or len(fwds) < f:
                continue
            xi = [best_gk] + defs[:d] + mids[:m] + fwds[:f]
            total = sum(xp(p, gw_onlook) for p in xi)
            if total > best_total_xp:
                best_total_xp = total
                best_eleven = xi
        if not best_eleven:
            all_sorted = sorted(self.players, key=lambda p: xp(p, gw_onlook), reverse=True)
            best_eleven = all_sorted[:11]
            best_total_xp = sum(xp(p, gw_onlook) for p in best_eleven)
        return best_eleven, best_total_xp