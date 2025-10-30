import pandas as pd
from typing import Dict
import numpy as np

FIELDS_TO_WEIGHT = ["assists", "goals_scored", "minutes", "goals_conceded",
                    "influence", "creativity", "threat", "ict_index", "bonus", "yellow_cards", "saves",
                    "expected_assists","expected_goal_involvements", "expected_goals", "expected_goals_conceded", "starts"]

INVERSE_FEATURES = ["penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"]

def load_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                path,
                engine="python",       # more tolerant
                on_bad_lines="skip",   # or "skip"
                encoding=enc
            )
        except UnicodeDecodeError:
            continue

    return pd.read_csv(
        path,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )


def get_path_from_row(root:str, row: pd.Series) -> str:
    first_name = row.get("first_name", "")
    last_name = row.get("second_name", "")
    id = row.get("id", "")
    return f"{root}/players/{first_name}_{last_name}_{id}"


def calculate_weighted_field(player_df, field_name, upcoming_gw, look_back=5):
    weighted_sum = 0.0
    previous_fixtures = (
        player_df[player_df["round"] < upcoming_gw]
        .sort_values(by="round", ascending=False)
        .head(look_back)
    )
    total_weight = 0.0
    for i, (_, fixture) in enumerate(previous_fixtures.iterrows()):
        weight = look_back - i  # i=0 (most recent) → biggest weight
        if fixture.get(field_name, None) is None:
            print(f"{field_name} not found")
        val = float(fixture.get(field_name, 0) or 0)
        weighted_sum += val * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def get_team_and_fixture_info(player_df, teams_df, upcoming_gw, fixtures_df, lookback=5):
    gw_fixtures = player_df[player_df["round"] == upcoming_gw]
    gw_data = [] # list of dicts for each fixture in the upcoming game week for that player (accounts for double gws)
    for _, fixture in gw_fixtures.iterrows():
        data = {}
        data["points"] = fixture["total_points"]
        opponent_team_id = fixture["opponent_team"]
        was_home = fixture["was_home"]
        data.update(opponent_performance_stats(fixtures_df, opponent_team_id, upcoming_gw, lookback))
        opponent_row = teams_df[teams_df["id"] == opponent_team_id]
        data["opposition_strength"] = opponent_row["strength"].values[0]
        if was_home:
            data["opponent_strength_attack"] = opponent_row["strength_attack_home"].values[0]
            data["opponent_strength_defense"] = opponent_row["strength_defence_home"].values[0]
        else:
            data["opponent_strength_attack"] = opponent_row["strength_attack_away"].values[0]
            data["opponent_strength_defense"] = opponent_row["strength_defence_away"].values[0]
        data["was_home"] = int(was_home)
        gw_data.append(data)
    
    return gw_data


def inverse_features(players_raw_row) -> Dict:
    data = {}
    for key in INVERSE_FEATURES:
        raw = players_raw_row.get(key, None)
        # Treat None/NaN/"" as missing
        if raw is None or (isinstance(raw, str) and raw.strip() == "") or pd.isna(raw):
            data[key] = 0.0
            continue
        # Coerce to float safely
        try:
            val = float(raw)
        except (TypeError, ValueError):
            data[key] = 0.0
            continue
        # Map ranks 1,2,3,... → 1.0, 0.5, 0.333... ; 0 or negatives → 0.0
        if not np.isfinite(val) or val <= 0:
            data[key] = 0.0
        else:
            data[key] = 1.0 / val
    return data


def opponent_performance_stats(fixtures_df, opponent_team_id, current_gw, lookback=5):
    df = fixtures_df[
        ((fixtures_df["team_a"] == opponent_team_id) | (fixtures_df["team_h"] == opponent_team_id))
        & (fixtures_df["event"] < current_gw)
    ]
    if "finished" in df.columns:
        df = df[df["finished"] == True]
    else:
        df = df[df["team_a_score"].notna() & df["team_h_score"].notna()]
    if df.empty:
        return {"opponent_goals_scored": 0.0, "opponent_goals_conceded": 0.0}

    recent = df.sort_values("event", ascending=False).head(lookback)

    goals_conceded = 0.0
    goals_scored = 0.0
    total_weight = 0.0

    for i, (_, fixture) in enumerate(recent.iterrows()):
        weight = lookback - i  # most recent → largest weight
        total_weight += weight
        if fixture["team_a"] == opponent_team_id:
            goals_conceded += float(fixture.get("team_h_score", 0) or 0) * weight
            goals_scored   += float(fixture.get("team_a_score", 0) or 0) * weight
        else:
            goals_conceded += float(fixture.get("team_a_score", 0) or 0) * weight
            goals_scored   += float(fixture.get("team_h_score", 0) or 0) * weight

    if total_weight == 0:
        return {"opponent_goals_scored": 0.0, "opponent_goals_conceded": 0.0}

    return {
        "opponent_goals_scored": goals_scored / total_weight,
        "opponent_goals_conceded": goals_conceded / total_weight,
    }


def get_team_and_fixture_info_current_season(upcoming_gw, teams_df, player_fixtures_df, fixtures_df):
    upcoming_gws = player_fixtures_df[player_fixtures_df["event"] == upcoming_gw]
    gw_data = [] # list of dicts for each fixture in the upcoming game week for that player (accounts for double gws)
    for _, fixture in upcoming_gws.iterrows():
        data = {}
        is_home = fixture["is_home"]
        opponent_team_id = fixture["team_a"] if is_home else fixture["team_h"]
        data.update(opponent_performance_stats(fixtures_df, opponent_team_id, upcoming_gw))
        opponent_row = teams_df[teams_df["id"] == opponent_team_id]
        data["opposition_strength"] = opponent_row["strength"].values[0]
        if is_home:
            data["opponent_strength_attack"] = opponent_row["strength_attack_home"].values[0]
            data["opponent_strength_defense"] = opponent_row["strength_defence_home"].values[0]
        else:
            data["opponent_strength_attack"] = opponent_row["strength_attack_away"].values[0]
            data["opponent_strength_defense"] = opponent_row["strength_defence_away"].values[0]
        data["was_home"] = int(is_home)
        gw_data.append(data)
    return gw_data


def position_dict(position:int) -> Dict:
    return {
        "goalkeeper": int(position == 1),
        "defender": int(position == 2),
        "midfielder": int(position == 3),
        "forward": int(position == 4)
    }


def prepare_training_data(outpath="../data/processed_player_data.csv", seasons=["2022-23", "2023-24", "2024-25"]):
    all_data = []
    for season in seasons:
        print("Processing season:", season)
        root = f"../../../data/Fantasy-Premier-League/data/{season}"
        players_raw_df = load_csv(f"{root}/players_raw.csv")
        teams_df = load_csv(f"{root}/teams.csv")
        fixtures_df = load_csv(f"{root}/fixtures.csv")
        for _, player_row in players_raw_df.iterrows():
            inverse_data = inverse_features(player_row)
            player_path = get_path_from_row(root, player_row)
            player_df = load_csv(f"{player_path}/gw.csv")
            position = player_row["element_type"]

            for i in range(5, 39):
                if player_row["minutes"] == 0:
                    continue
                data = {}
                for key in FIELDS_TO_WEIGHT:
                    data[key] = calculate_weighted_field(player_df, key, i)
                fixture_details = get_team_and_fixture_info(player_df, teams_df, i, fixtures_df)
                for fixture_dict in fixture_details:
                    combined_data = {**data, **fixture_dict}
                    combined_data["player_id"] = player_row["id"]
                    combined_data.update(position_dict(position))
                    combined_data["game_week"] = i
                    combined_data["season"] = season
                    combined_data.update(inverse_data)
                    all_data.append(combined_data)
    df = pd.DataFrame(all_data)
    df.to_csv(outpath, index=False)


def get_player_features_for_gw_current_season(upcoming_gw, teams_df, player_history_df, player_fixtures_df, players_raw_dict, all_fixtures_df):
    data = inverse_features(players_raw_dict)
    
    for key in FIELDS_TO_WEIGHT:
        data[key] = calculate_weighted_field(player_history_df, key, upcoming_gw)
    
    per_fixture_data = []
    fixture_details = get_team_and_fixture_info_current_season(upcoming_gw, teams_df, player_fixtures_df, all_fixtures_df)
    for fixture_dict in fixture_details:
        combined_data = {**data, **fixture_dict}
        position = players_raw_dict["element_type"]
        combined_data.update(position_dict(position))
        per_fixture_data.append(combined_data)
    return per_fixture_data



if __name__ == "__main__":
    prepare_training_data(outpath="../data/incl_set_pieces.csv")