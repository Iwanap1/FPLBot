from auto_fpl import XPPredictor, FPLDataFetcher

fetcher = FPLDataFetcher()
features = fetcher.featurize_all_players()
print(features.head())
predictor = XPPredictor("auto_fpl/models/20251028-111129-76fc2913")
top_n = predictor.top_n_players(features, n=30)
print(top_n)
for pid, xp in top_n:
    print(fetcher.id_to_name(pid), ":", xp)
