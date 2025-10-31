from .api_data_fetcher import FPLDataFetcher
from auto_fpl.team import Squad, Player
from auto_fpl.graph_traversal.transfer_history import TransferState
import time
from typing import List, Tuple


class AutoFPLBot:
    def __init__(self, xp_predictor, strategy_planner, fpl_controller=None, data_fetcher=None, squad=None):
        """Bot to automatically do your fpl team.
Args:
    xp_predictor: class containing a function "predict_xp" that takes pd.DataFrame of all features for every fixture of every player in that GW. 
    strategy_planner: class containing a function "transfer_strategy", with args all_players: List[Player] and starting_squad: Squad
    fpl_controller: class to obtain current squad information, submit transfers and organise squad
        """
        self.strategy_planner = strategy_planner
        if hasattr(self.strategy_planner, "depth"):
            self.depth = strategy_planner.depth
        else:
            self.depth = 1
        self.xp_predictor = xp_predictor
        self.fpl_controller = fpl_controller
        if data_fetcher is None:
            self.data_fetcher = FPLDataFetcher(cache_dfs=True)
        self.squad = squad


    def calculate_xps(self):
        all_xps = {pid: [] for pid in self.data_fetcher.all_player_ids}
        all_availibilities = {pid: self.data_fetcher.player_availavility(pid, onlook=self.depth) for pid in self.data_fetcher.all_player_ids}
        for i in range(self.depth):
            gw = self.data_fetcher.next_gameweek_id + i
            features = self.data_fetcher.featurize_all_players(gw)
            xps = self.xp_predictor.predict_xp(features)
            for pid, xp in xps.items():
                all_xps[pid].append(xp * all_availibilities[pid][i])

        self.data_fetcher.cached_player_dfs = {} # clear cache after use
        return all_xps
    
    
    def build_current_squad(self, squad_info=None):
        if squad_info is None:
            if self.fpl_controller is None:
                try:
                    from fpl_controller import FPLController
                    self.fpl_controller = FPLController()
                except Exception as e:
                    raise ValueError("FPLController is not provided and could not be imported.") from e
            squad_info = self.fpl_controller.get_current_squad_info()
        
        picks = squad_info['picks']
        pids = [int(p['element']) for p in picks]
        bank = float(squad_info["transfers"]["bank"]) / 10.0
        free_transfers = int(squad_info["transfers"]["limit"])
        players = [p for p in self.get_all_players() if p.pid in pids]
        price_overrides = {int(p["element"]): float(p.get("selling_price", p.get("purchase_price", 0))) / 10.0 for p in picks}

        for player in players:
            if player.pid in price_overrides:
                object.__setattr__(player, 'price', price_overrides[player.pid])
        
        squad = Squad(players=players, bank=bank, free_transfers=free_transfers)
        self.squad = squad
        self.og_squad = squad

        
    def get_all_players(self):
        players = []
        all_xps = self.calculate_xps()
        
        for p in self.data_fetcher.bootstrap_data['elements']:
            pid = p['id']
            price = p['now_cost'] / 10.0
            team_id = p['team']
            pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            pos = pos_map[p['element_type']]
            expected_xps = all_xps.get(pid, [0.0] * self.depth)
            player = Player(pid=pid, price=price, team_id=team_id, pos=pos, xps=expected_xps)
            players.append(player)
        self.all_players = players
        return players
    

    def decide_transfer_strategy(self, print_top_5=False):
        if self.squad is None:
            self.build_current_squad()
        strategies = self.strategy_planner.transfer_strategy(self.all_players, self.squad)

        if isinstance(strategies, list):
            strategies.sort(key=lambda x: x.accumilated_xp_gain, reverse=True)
            if print_top_5:
                self.print_strategies(strategies)
            self.strategies = strategies
            return strategies[0]

        elif isinstance(strategies, TransferState):
            if print_top_5:
                self.print_strategies([strategies])
                self.strategies = [strategies]
        return strategies
                

    def print_strategies(self, strategies):
        for i, strategy in enumerate(strategies[:5]):
            print("\n\n")
            print("Strategy Rank", i + 1)
            print("==========")
            for gw, transfers in enumerate(strategy.transfers_made):
                print(f"GW{self.data_fetcher.next_gameweek_id + gw}:")
                for transfer in transfers:
                    print(f"    OUT: {self.data_fetcher.id_to_name(transfer[0])} | IN: {self.data_fetcher.id_to_name(transfer[1])} ")
                print("\n")

    def do_team(self, commit=False):
        if not hasattr(self, "strategies"):
            self.decide_transfer_strategy()
        print("\nTRANSFERS")
        print("================")
        transfers = []
        for out_pid, in_pid in self.strategies[0].transfers_made[0]:
            print(f"OUT: {self.data_fetcher.id_to_name(out_pid)} | IN: {self.data_fetcher.id_to_name(in_pid)}")

            # find index in the actual squad by PID (not by Player equality)
            try:
                out_idx = next(i for i, p in enumerate(self.og_squad.players) if p.pid == out_pid)
            except StopIteration:
                raise ValueError(f"Outgoing player PID {out_pid} is not in current squad")
            in_player = next(p for p in self.all_players if p.pid == in_pid)
            out_player = next(p for p in self.all_players if p.pid == out_pid)
            self.og_squad.apply_swap(out_idx, in_player)
            transfers.append((out_player, in_player))
        if commit:
            "Print Committing Transfers..."
            self.commit_transfers(transfers)
            time.sleep(3)
            "Print Committing Squad..."
        self.manage_squad(commit)
        return
    

    def commit_transfers(self, transfers: List[Tuple[Player, Player]]):
        if self.fpl_controller is None:
            raise ValueError("No FPL Controller Provided")
        formatted_transfers = []
        for transfer in transfers:
            formatted_transfers.append({
                "element_in": int(transfer[1].pid),
                "element_out": int(transfer[0].pid),
                "purchase_price": int(transfer[1].price * 10),
                "selling_price": int(transfer[0].price * 10),
            })

        self.fpl_controller.submit_transfer(formatted_transfers)
        return
    
    def manage_squad(self, commit=False):
        best_xi, expected_points = self.og_squad.best_xi(0)
        best_xi.sort(key= lambda p: p.xps[0], reverse=True)
        captain = best_xi[0]
        vice_captain = best_xi[1]

        print("\nStarting XI")
        print("================")
        self.print_players(best_xi, captain, vice_captain)

        bench = [p for p in self.squad.players if p not in best_xi]
        bench.sort(key=lambda p: p.xps[0], reverse=True)

        print("\nBench")
        print("================")
        self.print_players(bench)

        plan = {"StartingXI": [p.pid for p in best_xi], "BenchOrder": [p.pid for p in bench], "Captain": captain.pid, "ViceCaptain": vice_captain.pid}
        if commit:
            self.fpl_controller.organise_squad(plan)


    def print_players(self, players: List[Player], captain=None, vice_captain=None):
        for position in ["GK", "DEF", "MID", "FWD"]:
            starters = [p for p in players if p.pos == position]
            for starter in starters:
                if starter == captain:
                    tail = "(C)"
                elif starter == vice_captain:
                    tail = "(VC)"
                else:
                    tail = ""
                print(f"{position}: {self.data_fetcher.id_to_name(starter.pid)} | xP = {round(starter.xps[0], 2)} {tail}")
