# scripts/06_paper_trading.py
"""
Step 06 — Paper Trading (Alpaca Paper + yfinance live data)
-----------------------------------------------------------
Wichtig: Entscheidet auf der letzten vollständig abgeschlossenen 1-Min Kerze (nicht auf der in-progress Kerze),
damit es mit Step 05 (Decision t, Execution t+1 Open) vergleichbar bleibt.

Praktisch live:
- Wir nehmen decision_bar = vorletzte Zeile aus yfinance-1m
- Order wird danach als Market Order gesendet (Proxy für "Open von t+1")
- Logs enthalten decision_time_utc und loop_ts (exec_time_utc)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import os
import time
import json
import requests

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"
    gold_symbol: str = "GLD"

    model_name: str = "logreg"  # logreg or rf

    entry_threshold: float = 0.55
    exit_threshold: float = 0.50
    max_holding_minutes: int = 15

    # optional extra hysteresis like backtest
    no_trade_margin: float = 0.00

    notional_usd: float = 1000.0

    poll_seconds: int = 60
    lookback_minutes: int = 180

    trade_only_when_market_open: bool = True

    base_dir: Path = PROJECT_ROOT

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "data" / "paper_trading"


class AlpacaPaperClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("APCA_TRADING_API_KEY_ID", "").strip()
        self.api_secret = os.getenv("APCA_TRADING_API_SECRET_KEY", "").strip()
        self.base_url = os.getenv("APCA_TRADING_API_BASE_URL", "https://paper-api.alpaca.markets").strip()

        if not self.api_key or not self.api_secret:
            raise RuntimeError("APCA_TRADING_API_KEY_ID / APCA_TRADING_API_SECRET_KEY fehlen (env/.env).")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def _get(self, path: str):
        url = self.base_url + path
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_orders(self, status: str = "open") -> list:
        """Return list of orders (default: open)."""
        return self._get(f"/v2/orders?status={status}&limit=50")

    def _post(self, path: str, payload: dict):
        url = self.base_url + path
        r = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()

    def get_clock(self) -> dict:
        return self._get("/v2/clock")

    def get_position(self, symbol: str) -> dict | None:
        try:
            return self._get(f"/v2/positions/{symbol}")
        except requests.HTTPError:
            return None

    def submit_market_order(self, symbol: str, side: str, qty: int, tif: str = "day") -> dict:
        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": tif,
        }
        return self._post("/v2/orders", payload)


def build_features_from_1m(spy_1m: pd.DataFrame, gld_1m: pd.DataFrame) -> pd.DataFrame:
    def normalize_and_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()

        # flatten MultiIndex columns -> "part1_part2"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(p) for p in col if p is not None and str(p).strip() != ""]).strip()
                for col in df.columns
            ]

        # ensure string columns, lowercase, replace spaces
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # add prefix (avoid double prefix)
        df = df.rename(columns={c: f"{prefix}_{c}" if not c.startswith(f"{prefix}_") else c for c in df.columns})

        cols = list(df.columns)

        # ensure a close column exists: prefer exact, then adj_close, then any containing 'close'
        target_close = f"{prefix}_close"
        close_candidates = [c for c in cols if c.endswith("_close") or "_close" in c]
        if close_candidates and target_close not in cols:
            chosen = next((c for c in close_candidates if "adj_close" in c), close_candidates[0])
            df = df.rename(columns={chosen: target_close})
            cols = list(df.columns)

        # ensure a volume column exists
        target_vol = f"{prefix}_volume"
        vol_candidates = [c for c in cols if c.endswith("_volume") or "_volume" in c]
        if vol_candidates and target_vol not in cols:
            df = df.rename(columns={vol_candidates[0]: target_vol})
            cols = list(df.columns)

        # final sanity: if critical cols missing -> clear error with available cols
        if target_close not in df.columns or target_vol not in df.columns:
            raise KeyError(
                f"Erwarte '{target_close}' und '{target_vol}' in den Spalten von {prefix}. "
                f"Gefundene Spalten: {cols}"
            )

        # remove duplicated column labels if any
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")]

        return df

    spy = normalize_and_prefix(spy_1m, "spy")
    gld = normalize_and_prefix(gld_1m, "gld")

    df = spy.join(gld, how="inner").sort_index()

    # ensure no duplicated index
    df = df[~df.index.duplicated(keep="last")]

    # --- Kompatibilität: erzeuge unpräfixierte Primärsymbol-Spalten (für Trainings-Feature-Namen) ---
    # Kopiere spy_* -> unprefixed (wenn noch nicht vorhanden)
    prim_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for col in prim_cols:
        dst = col
        src = f"spy_{col}"
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]

    # vwap fallback: wenn kein spy_vwap vorhanden, berechne Typical Price (high+low+close)/3
    if "vwap" not in df.columns:
        if all(c in df.columns for c in ("spy_high", "spy_low", "spy_close")):
            df["vwap"] = (df["spy_high"] + df["spy_low"] + df["spy_close"]) / 3.0
        elif all(c in df.columns for c in ("high", "low", "close")):
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # trade_count fallback -> 0
    if "trade_count" not in df.columns:
        df["trade_count"] = 0

    # Kopiere zentrale Feature-Namen vom prefixed- namespace (z. B. spy_ret_1m -> ret_1m)
    compat_map = {
        "ret_1m": "spy_ret_1m",
        "ret_5m": "spy_ret_5m",
        "ret_15m": "spy_ret_15m",
        "roll_mean_5m": "spy_roll_mean_5m",
        "roll_mean_15m": "spy_roll_mean_15m",
        "roll_std_5m": "spy_roll_std_5m",
        "roll_std_15m": "spy_roll_std_15m",
        "vol_roll_mean_15m": "spy_vol_roll_mean_15m",
        "vol_roll_std_15m": "spy_vol_roll_std_15m",
        "close_to_roll_mean_15m": "spy_close_to_roll_mean_15m",
        # ggf. weitere Mapping-Einträge hinzufügen
    }
    for dst, src in compat_map.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]

    # --- feature calculations (wie vorher) ---
    # Falls die spy_*-Spalten noch nicht existieren (z.B. bei anderen yfinance-Formaten), erzeugen wir sie oben.
    df["spy_ret_1m"] = df["spy_close"].pct_change(1)
    df["spy_ret_5m"] = df["spy_close"].pct_change(5)
    df["spy_ret_15m"] = df["spy_close"].pct_change(15)

    df["spy_roll_mean_5m"] = df["spy_close"].rolling(5).mean()
    df["spy_roll_mean_15m"] = df["spy_close"].rolling(15).mean()
    df["spy_roll_std_5m"] = df["spy_close"].rolling(5).std()
    df["spy_roll_std_15m"] = df["spy_close"].rolling(15).std()

    df["spy_vol_roll_mean_15m"] = df["spy_volume"].rolling(15).mean()
    df["spy_vol_roll_std_15m"] = df["spy_volume"].rolling(15).std()

    df["spy_close_to_roll_mean_15m"] = df["spy_close"] / df["spy_roll_mean_15m"] - 1

    df["gld_ret_1m"] = df["gld_close"].pct_change(1)
    df["gld_ret_5m"] = df["gld_close"].pct_change(5)
    df["gld_ret_15m"] = df["gld_close"].pct_change(15)

    df["gld_roll_mean_5m"] = df["gld_close"].rolling(5).mean()
    df["gld_roll_mean_15m"] = df["gld_close"].rolling(15).mean()
    df["gld_roll_std_5m"] = df["gld_close"].rolling(5).std()
    df["gld_roll_std_15m"] = df["gld_close"].rolling(15).std()

    df["gld_vol_roll_mean_15m"] = df["gld_volume"].rolling(15).mean()
    df["gld_vol_roll_std_15m"] = df["gld_volume"].rolling(15).std()

    df["gld_close_to_roll_mean_15m"] = df["gld_close"] / df["gld_roll_mean_15m"] - 1

    df["ret_spy_minus_gld_1m"] = df["spy_ret_1m"] - df["gld_ret_1m"]
    df["ret_spy_minus_gld_15m"] = df["spy_ret_15m"] - df["gld_ret_15m"]
    df["vol_ratio_spy_gld_15m"] = df["spy_roll_std_15m"] / (df["gld_roll_std_15m"] + 1e-12)
    df["price_ratio_spy_gld"] = df["spy_close"] / (df["gld_close"] + 1e-12)

    df["hour"] = df.index.hour
    df["minute_of_day"] = df["hour"] * 60 + df.index.minute
    df["minute_of_day_norm"] = df["minute_of_day"] / (24 * 60)

    return df


def feature_cols_from_train(train_df: pd.DataFrame, target_col: str = "target_up") -> list[str]:
    return [c for c in train_df.columns if c != target_col and not c.startswith("future_ret_")]


class LiveModel:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.model = None
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []

    def fit_from_train_csv(self, train_path: Path) -> None:
        train = pd.read_csv(train_path, index_col=0)
        self.feature_cols = feature_cols_from_train(train)

        X = train[self.feature_cols]
        y = train["target_up"].astype(int)

        if X.isnull().any().any():
            raise ValueError("NaNs in train.csv Features. Bitte Step 03 prüfen (dropna).")

        if self.cfg.model_name == "logreg":
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X)
            self.model = LogisticRegression(max_iter=800)
            self.model.fit(Xs, y)
        elif self.cfg.model_name == "rf":
            self.scaler = None
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X, y)
        else:
            raise ValueError("model_name must be 'logreg' or 'rf'")

    def predict_p_up(self, feat_row: pd.DataFrame) -> float:
        """
        Robust column alignment + scaling, returns probability for 'up'.
        Fehlende Features werden mit 0 aufgefüllt statt einen KeyError zu werfen.
        """
        import pandas as pd

        row = feat_row.copy()
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise ValueError("predict_p_up erwartet eine einzelne Zeile (DataFrame mit 1 Zeile).")
            row = row.iloc[0]

        available = list(feat_row.columns) if isinstance(feat_row, pd.DataFrame) else list(row.index)
        symbol = self.cfg.primary_symbol.lower()

        def clean_name(n: str) -> str:
            s = str(n).lower().strip()
            parts = [p for p in s.split("_") if p != ""]
            dedup = []
            for p in parts:
                if not dedup or p != dedup[-1]:
                    dedup.append(p)
            if len(dedup) > 1 and dedup[-1] == symbol and dedup.count(symbol) > 1:
                dedup.pop()
            return "_".join(dedup)

        cleaned_map = {clean_name(a): a for a in available}

        mapping: dict[str, str] = {}
        missing = []
        for f in self.feature_cols:
            if f in available:
                continue
            if f in cleaned_map:
                mapping[cleaned_map[f]] = f
                continue
            candidates = [a for a in available if a.lower().endswith(f)]
            if candidates:
                mapping[candidates[0]] = f
                continue
            candidates = [a for a in available if clean_name(a).endswith(f)]
            if candidates:
                mapping[candidates[0]] = f
                continue
            candidates = [a for a in available if f in a.lower() or f in clean_name(a)]
            if candidates:
                chosen = next((c for c in candidates if symbol in c.lower()), candidates[0])
                mapping[chosen] = f
                continue
            missing.append(f)

        if mapping:
            row = row.rename(mapping)

        # statt KeyError: fehlende Features mit 0.0 auffüllen (z.B. trade_count, vwap)
        not_found = [f for f in self.feature_cols if f not in row.index]
        if not_found:
            print(f"Warnung: fehlende Features {not_found} -> mit 0.0 aufgefüllt.")
            for f in not_found:
                row[f] = 0.0

        X = pd.DataFrame([row[self.feature_cols].astype(float).values], columns=self.feature_cols)

        if self.scaler is not None:
            Xs = pd.DataFrame(self.scaler.transform(X), columns=self.feature_cols)
        else:
            Xs = X

        if hasattr(self.model, "predict_proba"):
            p = float(self.model.predict_proba(Xs)[:, 1][0])
        else:
            p = float(self.model.predict(Xs)[0])

        return p


def yf_fetch_1m(symbol: str, lookback_minutes: int) -> pd.DataFrame:
    df = yf.download(symbol, period="1d", interval="1m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.tail(lookback_minutes)

    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def ensure_int_qty(notional_usd: float, last_price: float) -> int:
    if last_price <= 0:
        return 0
    return max(1, int(notional_usd / last_price))


def append_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if not log_path.exists():
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)


def effective_thresholds(cfg: ProjectConfig) -> tuple[float, float]:
    entry = float(cfg.entry_threshold)
    exit_ = float(cfg.exit_threshold)
    m = float(cfg.no_trade_margin)
    if m > 0:
        entry = max(entry, 0.5 + m)
        exit_ = min(exit_, 0.5 - m)
    if entry <= exit_:
        entry = max(entry, exit_ + 1e-6)
    return entry, exit_


def main():
    cfg = ProjectConfig()

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.logs_dir / f"paper_trading_{cfg.model_name}.csv"
    state_path = cfg.logs_dir / f"state_{cfg.model_name}.json"
    orders_path = cfg.logs_dir / f"orders_{cfg.model_name}.jsonl"

    alp = AlpacaPaperClient()

    train_path = cfg.processed_data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("train.csv fehlt. Bitte scripts/03_data_preparation.py ausführen.")

    live_model = LiveModel(cfg)
    live_model.fit_from_train_csv(train_path)

    state = {"in_position": False, "entry_time_utc": None}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    entry_eff, exit_eff = effective_thresholds(cfg)

    print(
        f"✅ Paper Trading gestartet | model={cfg.model_name} | "
        f"entry={entry_eff:.4f} exit={exit_eff:.4f} hold={cfg.max_holding_minutes}m | "
        f"alpaca_base={os.getenv('APCA_TRADING_API_BASE_URL', 'https://paper-api.alpaca.markets')}"
    )

    while True:
        loop_ts = datetime.now(timezone.utc)

        if cfg.trade_only_when_market_open:
            clock = alp.get_clock()

            # Debug: zeige clock details (UTC) — hilfreich für Timezone‑Verwirrung
            print(
                f"[DEBUG] loop_ts={loop_ts.isoformat()} clock_is_open={clock.get('is_open')} timestamp={clock.get('timestamp')} next_open={clock.get('next_open')} next_close={clock.get('next_close')}")

            if not clock.get("is_open", False) and not cfg.allow_trading_during_extended_hours:
                # zeige offene Orders zur Kontrolle (warum noch offene Order existiert)
                try:
                    open_orders = alp.get_orders("open")
                    print(f"[DEBUG] offene Orders count={len(open_orders)}")
                    for o in open_orders:
                        print(
                            f"[DEBUG] order id={o.get('id')} symbol={o.get('symbol')} status={o.get('status')} submitted_at={o.get('submitted_at')} filled_at={o.get('filled_at')}")
                except Exception as e:
                    print(f"[DEBUG] Fehler beim Abrufen von Orders: {e}")

                print(f"[{loop_ts.isoformat()}] Market closed -> sleep")
                time.sleep(cfg.poll_seconds)
                continue

        spy_1m = yf_fetch_1m(cfg.primary_symbol, cfg.lookback_minutes)
        gld_1m = yf_fetch_1m(cfg.gold_symbol, cfg.lookback_minutes)

        if spy_1m.empty or gld_1m.empty:
            print(f"[{loop_ts.isoformat()}] yfinance empty data -> sleep")
            time.sleep(cfg.poll_seconds)
            continue

        feat_df = build_features_from_1m(spy_1m, gld_1m).dropna()
        if len(feat_df) < 2:
            print(f"[{loop_ts.isoformat()}] not enough completed bars -> sleep")
            time.sleep(cfg.poll_seconds)
            continue

        # IMPORTANT: use last COMPLETED bar (second last row)
        decision_t = feat_df.index[-2]
        decision_row = feat_df.iloc[[-2]].copy()

        p_up = live_model.predict_p_up(decision_row)
        decision_price = float(decision_row["spy_close"].iloc[0])

        # alpaca truth
        pos = alp.get_position(cfg.primary_symbol)
        in_pos_api = pos is not None

        # reconcile state
        if in_pos_api and not state.get("in_position", False):
            state["in_position"] = True
            state["entry_time_utc"] = loop_ts.isoformat()
        if (not in_pos_api) and state.get("in_position", False):
            state["in_position"] = False
            state["entry_time_utc"] = None

        holding_min = 0
        if state.get("in_position") and state.get("entry_time_utc"):
            try:
                entry_t = datetime.fromisoformat(state["entry_time_utc"])
                holding_min = int((loop_ts - entry_t).total_seconds() / 60.0)
            except Exception:
                holding_min = 0

        action = "HOLD"
        order_resp = None

        if not in_pos_api:
            if p_up >= entry_eff:
                qty = ensure_int_qty(cfg.notional_usd, decision_price)
                if qty > 0:
                    order_resp = alp.submit_market_order(cfg.primary_symbol, "buy", qty)
                    action = f"BUY qty={qty}"
                    state["in_position"] = True
                    state["entry_time_utc"] = loop_ts.isoformat()
        else:
            if (p_up <= exit_eff) or (holding_min >= cfg.max_holding_minutes):
                qty = int(float(pos.get("qty", 0))) if pos else 0
                if qty > 0:
                    order_resp = alp.submit_market_order(cfg.primary_symbol, "sell", qty)
                    action = f"SELL qty={qty}"
                    state["in_position"] = False
                    state["entry_time_utc"] = None

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        log_row = {
            "exec_time_utc": loop_ts.isoformat(),
            "decision_time_utc": str(decision_t),
            "p_up": float(p_up),
            "decision_spy_close": float(decision_price),
            "in_position_api": int(in_pos_api),
            "holding_minutes": int(holding_min),
            "entry_threshold_effective": float(entry_eff),
            "exit_threshold_effective": float(exit_eff),
            "action": action,
        }
        append_log(log_path, log_row)

        print(f"[{loop_ts.isoformat()}] decision_t={decision_t} p_up={p_up:.4f} pos={in_pos_api} hold={holding_min} -> {action}")

        if order_resp is not None:
            with open(orders_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"exec_time_utc": loop_ts.isoformat(), "order": order_resp}) + "\n")

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()