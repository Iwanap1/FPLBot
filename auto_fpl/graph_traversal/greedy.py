from typing import List, Tuple
from copy import deepcopy
from tqdm import tqdm
from auto_fpl.team import Player, Squad
from .transfer_history import TransferState


class GreedyTransferPlanner:
    def __init__(
        self,
        depth: int = 8,
        min_bank_for_double: float = 4.0,
        double_transfers_to_evaluate: int = 10,
        transfer_penalty: int = 4,
    ):
        """
        Greedy planner: at each GW pick the single best action (skip / 1 xfer / 2 xfers)
        that maximises the current GW best XI xP minus hits, then move on.
        """
        self.depth = depth
        self.min_bank_for_double = min_bank_for_double
        self.double_transfers_to_evaluate = double_transfers_to_evaluate
        self.transfer_penalty = transfer_penalty
        self.gw_onlook = 0  # increments each loop

    # ---------- Public API ----------

    def transfer_strategy(self, all_players: List[Player], starting_squad: Squad) -> TransferState:
        """
        Returns a single TransferState representing the greedy path across `depth` weeks.
        """
        self.all_players = all_players
        self.gw_onlook = 0

        # Initialise state (no transfers yet)
        ft_start0 = starting_squad.free_transfers
        init_score = self._score_gw(starting_squad, ft_start0, used_this_gw=0)
        current = TransferState(
            current_squad=deepcopy(starting_squad),
            transfers_made=[[] for _ in range(self.depth)],
            accumilated_xp_gain=init_score,
        )

        pbar = tqdm(range(self.depth), desc="Greedy traversal", dynamic_ncols=True)
        for _ in pbar:
            before = current.accumilated_xp_gain
            current = self._expand_one_week(current)
            after = current.accumilated_xp_gain
            pbar.set_postfix(gw=self.gw_onlook, gain=f"{after - before:.2f}")
            self._advance_one_week(current)

        return current

    def _expand_one_week(self, state: TransferState) -> TransferState:
        """
        From `state`, evaluate:
          - Skip
          - All single swaps
          - Greedy doubles (second swap chosen by local ranking)
        Return the single best child state for this GW.
        """
        ft_start = state.current_squad.free_transfers
        print(ft_start)
        base_acc = state.accumilated_xp_gain
        best_state = deepcopy(state)
        skip_gain = self._score_gw(best_state.current_squad, ft_start, used_this_gw=0)
        best_state.accumilated_xp_gain = base_acc + skip_gain

        single_children: List[TransferState] = []
        for out_idx, out_p in enumerate(state.current_squad.players):
            for cand in self.all_players:
                if cand.pid in state.current_squad.all_pids:
                    continue
                if not state.current_squad.can_swap(out_idx, cand):
                    continue
                child = deepcopy(state)
                child.current_squad.apply_swap(out_idx, cand)
                child.current_squad.free_transfers -= 1
                gain = self._score_gw(child.current_squad, ft_start, used_this_gw=1)
                child.accumilated_xp_gain = base_acc + gain
                child.transfers_made[self.gw_onlook] = [(out_p.pid, cand.pid)]
                single_children.append(child)
                if ft_start >= 2 or child.current_squad.bank >= self.min_bank_for_double:
                    doubles = self._best_doubles_from_single(child, ft_start, base_acc)
                    single_children.extend(doubles)
        candidates = [best_state] + single_children
        if not candidates:
            return best_state

        return max(candidates, key=lambda s: s.accumilated_xp_gain)

    def _best_doubles_from_single(self, state_after_single: TransferState, ft_start: int, base_acc: float) -> List[TransferState]:
        squad = state_after_single.current_squad
        candidate_transfers: List[Tuple[Player, Player, float]] = []
        for out_idx, out_p in enumerate(squad.players):
            for cand in self.all_players:
                if cand.pid in squad.all_pids:
                    continue
                if not squad.can_swap(out_idx, cand):
                    continue
                delta = self._compare_players(cand, out_p)
                candidate_transfers.append((out_p, cand, delta))

        candidate_transfers.sort(key=lambda t: t[2], reverse=True)
        candidate_transfers = candidate_transfers[: self.double_transfers_to_evaluate]

        best_children = []
        for out_p, cand, _ in candidate_transfers:
            child = deepcopy(state_after_single)
            out_idx = child.current_squad.players.index(out_p)
            child.current_squad.apply_swap(out_idx, cand)
            child.current_squad.free_transfers -= 1
            gain = self._score_gw(child.current_squad, ft_start, used_this_gw=2)
            child.accumilated_xp_gain = base_acc + gain
            child.transfers_made[self.gw_onlook].append((out_p.pid, cand.pid))
            best_children.append(child)

        return best_children

    def _compare_players(self, candidate: Player, out_player: Player) -> float:
        return candidate.xps[self.gw_onlook] - out_player.xps[self.gw_onlook]

    def _score_gw(self, squad: Squad, ft_start: int, used_this_gw: int) -> float:
        """Best XI xP for this GW minus hits based on ft_start and used transfers."""
        hits = self.transfer_penalty * max(0, used_this_gw - ft_start)
        xi_xp = squad.best_xi(self.gw_onlook)[1]
        return xi_xp - hits

    def _advance_one_week(self, state: TransferState) -> None:
        self.gw_onlook += 1
        state.current_squad.free_transfers = min(state.current_squad.free_transfers + 1, 5)
        if state.current_squad.free_transfers < 1:
            state.current_squad.free_transfers = 1
