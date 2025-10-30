from auto_fpl import AutoFPLBot, HeuristicBeamSearchTransferPlanner, FPLDataFetcher, FPLController, XPPredictor, GreedyTransferPlanner
import os, json


# FPL controller to get current squad info and optionally submit transfers/squad. Alternatively, provide squad info to AutoFPLBot().build_current_squad()
# Can set to None if you dont want these features, but ensure you provide squad_info to AutoFPLBot().build_current_squad()
# With a controller, squad will be build automatically
controller = FPLController(
    email=os.getenv("EMAIL"),
    password=os.getenv("PASSWORD"),
    team_id=os.getenv("FPL_ID")
)


xp_predictor = XPPredictor(model_dir="auto_fpl/models/20251029-163004-3b0a27a3")


# TRANSFER PLANNERS 
beam_planner = HeuristicBeamSearchTransferPlanner(
    depth=5,                            # how far down the transfer tree to traverse
    min_bank_for_double=4,              # how much money in the bank after a transfer to consider a second transfer with only 1 free transfer
    double_transfers_to_evaluate=10,    # max no. of double transfers to consider for each single transfer. Beware of combinatorial explosion
    short_term_horizon=3,               # how many gameweeks to consider in deciding what immediate transfers to consider
    beam_width=20,                      # how many transfer strategies progress to the next depth
    transfer_penalty=4,                 # default FPL but can change if you want to increase/decrease how risky the strategy planner is with point hits
    always_consider_skip=True,          # ensures skipping the first transfer will always progress to the next layer
    min_avg_points=2.6                  # a player needs to average more than this xP over the short_term_horizon to be considered for transfer
)

greedy_planner = GreedyTransferPlanner(
    depth = 5,
    min_bank_for_double=4,
    transfer_penalty=4
)

with open("test_squad.json", "r") as f:
    test_squad = json.load(f)


fetcher = FPLDataFetcher(cache_dfs=True)

bot = AutoFPLBot(
    xp_predictor=xp_predictor,
    strategy_planner=beam_planner, # choose transfer planner
    fpl_controller=controller
)
bot.build_current_squad(test_squad)
bot.decide_transfer_strategy(print_top_5=True)