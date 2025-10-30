from auto_fpl.team import Player, Squad
from typing import List
from .transfer_history import TransferState
from copy import deepcopy
from tqdm import tqdm

class HeuristicBeamSearchTransferPlanner:
    def __init__(
        self,
        depth: int = 8,
        min_bank_for_double: float = 4.0,
        double_transfers_to_evaluate: int = 10,
        short_term_horizon: int = 3,
        beam_width: int = 5,
        transfer_penalty = 4,
        always_consider_skip: bool = True,
        min_avg_points: float = 2.0
    ):
        """Beam search transfer planner using heuristics to limit branching.
Args:
    starting_squad (Squad): The initial squad to start the graph from.
    depth (int): The depth of the graph (number of transfer windows to consider).
    min_bank_for_double (float): If the state has a certain amount in the bank, consider a double transfer
    short_term_horizon (int): Number of gameweeks to consider for short-term XP gain when evaluating transfers.
    beam_width (int): Number of top nodes to keep at each level of the graph.
    transfer_penalty (int): Points deducted for each transfer made beyond free transfers, default is 4 as per FPL but can shrink to encourage double transfers.
    always_consider_skip (bool): set to true to ensure a skip is always considered in the next layer of every branch. Will effectively double the beam_width beyond the first layer.
    min_avg_points (float): the minimum number of points a player can average over the short term horizon to be considered for a transfer 
"""
        
        self.gw_onlook = 0 # current depth
        self.depth = depth
        self.min_bank_for_double = min_bank_for_double
        self.double_transfers_to_evaluate = double_transfers_to_evaluate
        self.short_term_horizon = short_term_horizon
        self.beam_width = beam_width
        self.transfer_penalty = transfer_penalty
        self.always_consider_skip = always_consider_skip
        self.min_avg_points = min_avg_points
        
    
    def transfer_strategy(self, all_players: List[Player], starting_squad: Squad) -> List[TransferState]:
        print("Traversing Possible Transfers...")
        self.all_players = all_players
        self.squad = starting_squad
        pbar = tqdm(range(self.depth), desc="Traversal", dynamic_ncols=True)
        for i in pbar:
            if i == 0:
                states = self.first_expand()
                kept_before = len(states)
                states = self.shrink_states(states)
                kept_after = len(states)
                states = self.next_gameweek(states)
            else:
                expanded = self.expand_states(states)
                kept_before = sum(len(g) for g in expanded)
                states = self.shrink_states(expanded)
                kept_after = len(states)
                states = self.next_gameweek(states)
            pbar.set_postfix(expanded=kept_before, kept=kept_after)
        return states

    def _filter_candidates(self):
        players = self.all_players
        horizon = min(self.short_term_horizon, self.depth - self.gw_onlook)
        if horizon <= 0:
            return players  # end of horizon; don't filter
        end = self.gw_onlook + horizon
        return [
            p for p in players
            if (sum(p.xps[self.gw_onlook:end]) / float(horizon)) > self.min_avg_points
        ]

    def _compare_players(self, candidate: Player, out_player: Player) -> float:
        return sum(candidate.xps[self.gw_onlook:self.gw_onlook + self.short_term_horizon]) - sum(out_player.xps[self.gw_onlook:self.gw_onlook + self.short_term_horizon])

    def shrink_states(self, states: List[TransferState]) -> List[TransferState]:
        """keeps the top beam_width states based on the accumulated xp gain and transfer hits. Ensure skip is first in list."""
        if self.gw_onlook == 0:
            final_states = []
            if self.always_consider_skip:
                final_states.append(states[0])
                remove_skip = states[1:]
                remove_skip.sort(key=lambda s: s.accumilated_xp_gain, reverse=True)
                final_states.extend(remove_skip[:self.beam_width - 1])
            else:
                states.sort(key=lambda s: s.accumilated_xp_gain, reverse=True)
                final_states.extend(states[:self.beam_width])
            return final_states
        else:
            final_states = []
            flattened = []
            for state_set in states:
                flattened.extend(state_set)
            flattened.sort(key=lambda s: s.accumilated_xp_gain, reverse=True)
            final_states.extend(flattened[:self.beam_width])
            return final_states
            

    def next_gameweek(self, transfer_states: list[TransferState]) -> list[TransferState]:
        self.gw_onlook += 1
        for state in transfer_states:
            state.current_squad.free_transfers = min(state.current_squad.free_transfers + 1, 5)
            if state.current_squad.free_transfers < 1:
                state.current_squad.free_transfers = 1
        return transfer_states


    def first_expand(self) -> list[TransferState]:
        skip_squad = deepcopy(self.squad)
        ft_start0 = self.squad.free_transfers
        original_xp = self._calculate_gw_best_xi_xp(self.squad, ft_start0, 0)
        skip_state = TransferState(
            current_squad=skip_squad,
            transfers_made=[[] for _ in range(self.depth)],
            accumilated_xp_gain=original_xp
        )
        transfer_states = [skip_state]
        for i, player in enumerate(self.squad.players):
            for candidate in self._filter_candidates():
                if not candidate.pid in self.squad.all_pids and self.squad.can_swap(i, candidate):
                    new_squad = self._clone_with_swap(self.squad, i, candidate)
                    new_squad.free_transfers -= 1
                    xp = self._calculate_gw_best_xi_xp(new_squad, ft_start0, 1)
                    transfers_made = [[] for _ in range(self.depth)]
                    transfers_made[0] = [(player.pid, candidate.pid)]
                    state = TransferState(
                        current_squad=new_squad,
                        transfers_made=transfers_made,
                        accumilated_xp_gain=xp
                    )
                    transfer_states.append(state)
                    if new_squad.bank >= self.min_bank_for_double or ft_start0 > 1:
                        transfer_states.extend(self.find_double_transfers(state, 0.0, ft_start0))
        return transfer_states


    def find_double_transfers(self, transfer_state: TransferState, original_xp: float, original_free_transfers: int) -> list[TransferState]:
        """"find top 'double_transfers_to_evaluate' greedy second transfers for a given transfer state"""
        transfer_states = []
        candidate_transfers = []
        for player in transfer_state.current_squad.players:
            candidates = [p for p in self.all_players if p.pid not in transfer_state.current_squad.all_pids and transfer_state.current_squad.can_swap(transfer_state.current_squad.players.index(player), p)]
            for candidate in candidates:
                candidate_transfers.append((player, candidate))
        
        candidate_transfers.sort(key=lambda c: self._compare_players(c[1], c[0]), reverse=True)
        for candidate in candidate_transfers:
            new_state = deepcopy(transfer_state)
            new_state.current_squad.apply_swap(new_state.current_squad.players.index(candidate[0]), candidate[1])
            new_state.current_squad.free_transfers -= 1
            xp = original_xp + self._calculate_gw_best_xi_xp(new_state.current_squad, original_free_transfers, 2)
            new_state.accumilated_xp_gain = xp
            new_state.transfers_made[self.gw_onlook].append((candidate[0].pid, candidate[1].pid))
            if new_state not in transfer_states:
                transfer_states.append(new_state)
            if len(transfer_states) == self.double_transfers_to_evaluate:
                return transfer_states
        return transfer_states
        
    
    def _calculate_gw_best_xi_xp(self, squad: Squad, ft_start: int, used_this_gw: int):
        hits = self.transfer_penalty * max(0, used_this_gw - ft_start)
        score = squad.best_xi(self.gw_onlook)[1] - hits
        return score


    def expand_states(self, states: List[TransferState]):
        all_states = []
        for state in states:
            # transfers_made = deepcopy(state.transfers_made)
            original_xp = state.accumilated_xp_gain
            ft_start_0 = state.current_squad.free_transfers
            skip_state = deepcopy(state)
            new_xp = self._calculate_gw_best_xi_xp(skip_state.current_squad, ft_start_0, 0)
            skip_state.accumilated_xp_gain = original_xp + new_xp
            new_states = [skip_state]
            for i, player in enumerate(state.current_squad.players):
                for candidate in self._filter_candidates():
                    if not candidate.pid in state.current_squad.all_pids and state.current_squad.can_swap(i, candidate):
                        new_squad = self._clone_with_swap(state.current_squad, i, candidate)
                        new_squad.free_transfers -= 1
                        xp = self._calculate_gw_best_xi_xp(new_squad, ft_start_0, 1)
                        new_transfers_made = [lst.copy() for lst in state.transfers_made]
                        new_transfers_made[self.gw_onlook] = [(player.pid, candidate.pid)]
                        new_state = TransferState(
                            current_squad=new_squad,
                            transfers_made=new_transfers_made,
                            accumilated_xp_gain=original_xp + xp
                        )
                        new_states.append(new_state)
                        if new_squad.bank >= self.min_bank_for_double or ft_start_0 > 1:
                            new_states.extend(self.find_double_transfers(new_state, original_xp, ft_start_0))
            all_states.append(new_states)
        return all_states

    def _clone_with_swap(self, squad: Squad, out_idx: int, incoming: Player) -> Squad:
        # minimal clone
        new_players = list(squad.players)
        out_p = new_players[out_idx]
        new_players[out_idx] = incoming
        new_bank = squad.bank - (incoming.price - out_p.price)
        new_squad = Squad.__new__(Squad)    # allocate without __init__
        new_squad.players = new_players
        new_squad.bank = new_bank
        new_squad.free_transfers = squad.free_transfers
        new_squad.all_pids = (squad.all_pids - {out_p.pid}) | {incoming.pid}
        return new_squad
                        

    
                        



