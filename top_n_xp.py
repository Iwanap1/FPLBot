from auto_fpl import XPPredictor, FPLDataFetcher


predictor = XPPredictor("auto_fpl/models/20251030-143731-e772d071")
fetcher = FPLDataFetcher()
features = fetcher.featurize_all_players()
print(features.head())
top_n = predictor.top_n_players(features, n=30)
print(top_n)
for pid, xp in top_n:
    print(fetcher.id_to_name(pid), ":", xp)
