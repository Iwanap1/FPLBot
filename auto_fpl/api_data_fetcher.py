import requests
import pandas as pd
from auto_fpl.xp_prediction.prep_data import get_player_features_for_gw_current_season
from tqdm import tqdm
import re

class FPLDataFetcher:
    def __init__(self, cache_dfs=True):
        self.cache = cache_dfs
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.bootstrap_data = requests.get(self.base_url + 'bootstrap-static').json()
        self.next_gameweek_id = self.next_gameweek()
        self.cached_player_dfs = {}
        self.all_player_ids = [p["id"] for p in self.bootstrap_data['elements']]
        self.all_fixtures_df = pd.DataFrame(requests.get(self.base_url + 'fixtures').json())

    def next_gameweek(self):
        for event in self.bootstrap_data['events']:
            if event["is_next"]:
                return event["id"]
        return None


    def player_name_to_id(self, player_name: str) -> int:
        for player in self.bootstrap_data['elements']:
            full_name = f"{player['first_name']} {player['second_name']}"
            if full_name.lower() == player_name.lower():
                return player['id']
        raise ValueError(f"Player '{player_name}' not found.")
    
    def id_to_name(self, player_id: int) -> str:
        for player in self.bootstrap_data['elements']:
            if player['id'] == player_id:
                return f"{player['first_name']} {player['second_name']}"
        raise ValueError(f"Player ID '{player_id}' not found.")


    def fetch_player_dfs(self, player):
        if isinstance(player, str):
            player = self.player_name_to_id(player)
        assert player > 0, "Player ID must be a positive integer or a valid player name"

        if player in self.cached_player_dfs:
            return self.cached_player_dfs[player]
        
        player_url = f"{self.base_url}element-summary/{player}/"
        player_data = requests.get(player_url).json()
        history_df = pd.DataFrame(player_data['history'])
        fixtures_df = pd.DataFrame(player_data['fixtures'])
        if self.cache:
            self.cached_player_dfs[player] = (history_df, fixtures_df)
        return history_df, fixtures_df
    

    def get_player_current_data(self, player) -> int:
        player_id = self.player_name_to_id(player) if isinstance(player, str) else player
        for player in self.bootstrap_data['elements']:
            if player['id'] == player_id:
                return player
        raise ValueError(f"Player ID '{player_id}' not found.")
    

    def featurize_player(self, player, gw):
        pid = self.player_name_to_id(player) if isinstance(player, str) else player
        teams_df = pd.DataFrame(self.bootstrap_data['teams'])
        player_data = self.get_player_current_data(pid)
        position = player_data['element_type']

        try:
            history_df, fixtures_df = self.cached_player_dfs[pid]
        except KeyError:
            history_df, fixtures_df = self.fetch_player_dfs(pid)
        
        return get_player_features_for_gw_current_season(gw, teams_df, history_df, fixtures_df, position, self.all_fixtures_df)
    

    def featurize_all_players(self, gw=None) -> pd.DataFrame:
        if gw is None:
            gw = self.next_gameweek_id
        all_features = []
        for i, pid in enumerate(tqdm(self.all_player_ids, desc=f"Featurising players for GW{gw}")):
            try:
                features = self.featurize_player(pid, gw)
                for feature in features:
                    feature['pid'] = pid
                    all_features.append(feature)
            except Exception as e:
                print(f"Error featurizing player ID {pid}: {e}")
                continue

        return pd.DataFrame(all_features)
    
    def player_availavility(self, pid, start_gw, onlook=1):
        """returns a vector of 0 or 1 for availability of each onlook GW"""
        player_data = self.get_player_current_data(pid)
        availability = [1 for _ in range(onlook)]
        news = player_data.get("news", None)
        if news is None or news == "":
            return availability
        if "unknown" in news.lower():
            return [0 for _ in range(onlook)]
        try:
            history_df, fixtures_df = self.cached_player_dfs[pid]
        except KeyError:
            history_df, fixtures_df = self.fetch_player_dfs(pid)
        
        fixtures = fixtures_df[(fixtures_df["event"] >= start_gw) & (fixtures_df["event"] < start_gw + onlook)]
        months = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        date = re.compile(rf"\b(?P<day>\d+)\s*(?P<month>{months})", flags=re.I)
        m = date.search(news)
        if not m:
            return [0 for _ in range(onlook)]
    
        return_day = m.group("day")
        return_month = m.group("month")
        

    def player_availavility(self, pid, start_gw=None, onlook=1):
        """Return a list[int] of length `onlook`, 1 if player is likely available to play in that GW, else 0.
        Uses player's news string + per-player upcoming fixtures kickoff times.
        """
        start_gw = self.next_gameweek_id if start_gw is None else start_gw
        player_id = self.player_name_to_id(pid) if isinstance(pid, str) else pid
        player_row = self.get_player_current_data(player_id)

        availability = [1 for _ in range(onlook)]
        news = player_row.get("news", "") or ""
        news_l = news.lower().strip()
        
        if news_l == "":
            return availability
        if "unknown" in news_l:
            return [0 for _ in range(onlook)]
        if "rest of the season" in news_l:
            return [0 for _ in range(onlook)]
        if "on loan" in news_l:
            return [0 for _ in range(onlook)]
        if re.search(r"\bjoined\s.*\spermanently\b", news_l):
            return [0 for _ in range(onlook)]
        
        try:
            _, fixtures_df = self.cached_player_dfs[player_id]
        except KeyError:
            _, fixtures_df = self.fetch_player_dfs(player_id)

        # Parse "Expected back 22 Nov" (or similar). We also accept "back 22 Nov".
        months_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        m = re.search(r"\b(?:expected\s+back|back)\s+(\d{1,2})\s+([a-z]{3})", news_l)
        expected_dt = None
        if m is not None:
            day = int(m.group(1))
            mon_abbr = m.group(2)[:3]
            month = months_map.get(mon_abbr, None)

            if month is not None:
                gws = fixtures_df[
                    (fixtures_df.get("event").notna()) &
                    (fixtures_df["event"] >= start_gw)
                ].copy()
                kickoff_year = None
                if "kickoff_time" in gws.columns and gws["kickoff_time"].notna().any():
                    try:
                        earliest_kick = pd.to_datetime(gws["kickoff_time"], errors="coerce").min()
                        if pd.notna(earliest_kick):
                            kickoff_year = int(earliest_kick.year)
                    except Exception:
                        kickoff_year = None

                if kickoff_year is None:
                    kickoff_year = pd.Timestamp.utcnow().year
                if month <= 6 and 8 <= pd.Timestamp(kickoff_year, 1, 1).month:
                    pass  
                if "earliest_kick" in locals() and pd.notna(earliest_kick):
                    ek_mon = earliest_kick.month
                    exp_year = kickoff_year
                    if ek_mon >= 8 and month <= 6:
                        exp_year = kickoff_year + 1
                else:
                    now_mon = pd.Timestamp.utcnow().month
                    exp_year = kickoff_year + (1 if (now_mon >= 8 and month <= 6) else 0)

                try:
                    expected_dt = pd.Timestamp(exp_year, month, day, tz="UTC")
                except ValueError:
                    expected_dt = None

        if expected_dt is not None:
            fx = fixtures_df.copy()
            if "kickoff_time" in fx.columns:
                fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"], errors="coerce", utc=True)
            else:
                fx["kickoff_time"] = pd.NaT

            for i in range(onlook):
                gw = start_gw + i
                gw_rows = fx[fx.get("event") == gw]
                if not gw_rows.empty and gw_rows["kickoff_time"].notna().any():
                    earliest = gw_rows["kickoff_time"].min()
                    if pd.notna(earliest) and earliest < expected_dt:
                        availability[i] = 0
                    else:
                        availability[i] = 1
                else:
                    availability[i] = 1
            return availability

        # If no pattern match, assume available
        return availability
