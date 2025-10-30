from auto_fpl import XPPredictor, FPLDataFetcher

fetcher = FPLDataFetcher()
features = fetcher.featurize_all_players()
print(features.head())
predictor = XPPredictor("auto_fpl/models/20251029-163004-3b0a27a3")
top_n = predictor.top_n_players(features, n=10)
print(top_n)
for pid, xp in top_n:
    print(fetcher.id_to_name(pid), ":", xp)
