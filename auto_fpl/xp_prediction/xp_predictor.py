import torch
import joblib
import json
import pandas as pd
from .model import XPModel

class XPPredictor:
    def __init__(self, model_dir: str):
        self.model, self.scaler, self.feature_columns = self.load_xp_model(model_dir)

    # def load_xp_model(self, model_dir):
    #     features_path = f"{model_dir}/features.json"
    #     scaler_path   = f"{model_dir}/scaler.pkl"
    #     model_path    = f"{model_dir}/model.pt"

    #     with open(features_path, "r") as f:
    #         feature_columns = json.load(f)
    #     scaler = joblib.load(scaler_path)

    #     model = XPModel(input_size=len(feature_columns))
    #     state = torch.load(model_path, map_location="cpu")
    #     model.load_state_dict(state)
    #     model.eval()

    #     return model, scaler, feature_columns


    def _infer_hidden_list_from_state(self, state_dict):
        """
        Infer hidden sizes from new-style 'layers.*' checkpoint.
        Only consider 2-D weights (Linear layers). BN weights are 1-D -> skip.
        """
        linear_layers = []
        for k, v in state_dict.items():
            if not (k.startswith("layers.") and k.endswith(".weight")):
                continue
            try:
                shape = tuple(v.shape)
            except Exception:
                continue
            if len(shape) != 2:
                continue
            idx = int(k.split(".")[1])
            out_f, in_f = shape
            linear_layers.append((idx, out_f, in_f))

        if not linear_layers:
            return None
        linear_layers.sort(key=lambda t: t[0])
        outs = [o for _, o, _ in linear_layers]
        if outs and outs[-1] == 1:
            outs = outs[:-1]
        return outs if outs else None


    def load_xp_model(self, model_dir):
        features_path = f"{model_dir}/features.json"
        scaler_path   = f"{model_dir}/scaler.pkl"
        model_path    = f"{model_dir}/model.pt"

        with open(features_path, "r") as f:
            feature_columns = json.load(f)
        scaler = joblib.load(scaler_path)
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        if any(k.startswith(("fc1", "fc2", "fc3")) for k in state.keys()):
            model = XPModel(input_size=len(feature_columns), hidden_size=64)
        else:
            hidden = self._infer_hidden_list_from_state(state) or [64, 32]
            model = XPModel(input_size=len(feature_columns), hidden_size=hidden)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[XPPredictor] load_state_dict: missing={missing}, unexpected={unexpected}")
        model.eval()
        return model, scaler, feature_columns


    def predict_xp(
        self,
        player_features: pd.DataFrame,
        pid_col: str = "pid",
        return_per_fixture: bool = False,
    ):
        """
        Predict per-row xP and aggregate by pid (sum).
        If return_per_fixture=True, also return the per-row predictions DataFrame.
        Returns:
            dict {pid: summed_xP}
        """
        # sanity checks
        if pid_col not in player_features.columns:
            raise ValueError(f"'{pid_col}' column is required in player_features.")

        missing = set(self.feature_columns) - set(player_features.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")

        X = player_features[self.feature_columns].to_numpy(copy=False)
        X_scaled = self.scaler.transform(X)

        X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32)
        with torch.inference_mode():
            preds = self.model(X_tensor).squeeze(-1).cpu().numpy()

        per_row = player_features[[pid_col]].copy()
        per_row["xP"] = preds
        by_player = per_row.groupby(pid_col, sort=False)["xP"].sum()

        if return_per_fixture:
            return by_player.to_dict(), per_row
        return by_player.to_dict()


    def top_n_players(self, player_features: pd.DataFrame, n: int = 10, pid_col: str = "pid"):
        """
        Return the top-n players for the upcoming GW as a list of (pid, xP),
        sorted by predicted xP descending. Aggregates per-fixture rows per pid.
        """
        if n <= 0:
            return []

        # Reuse the validated prediction path (checks pid_col and feature columns)
        xp_by_pid = self.predict_xp(player_features, pid_col=pid_col, return_per_fixture=False)
        # xp_by_pid is a dict {pid: summed_xP}; sort and take top-n
        top = sorted(xp_by_pid.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:n]
        return top
