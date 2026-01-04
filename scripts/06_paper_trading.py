# scripts/06_paper_trading.py
"""
Step 06 — Paper Trading (Alpaca Paper + yfinance live data)
-----------------------------------------------------------
Ziele:
- Live(ish) 1-Min Daten via yfinance (SPY + GLD) ziehen
- Features wie in scripts/03_data_preparation.py berechnen
- Modell (LogReg oder RF) auf train.csv fitten
- Live-Signal p_up berechnen
- Trading-Regeln anwenden:
  - Entry: p_up >= entry_threshold (wenn keine Position)
  - Exit:  p_up <= exit_threshold ODER max_holding_minutes erreicht
- Orders über Alpaca PAPER API platzieren und Positionen tracken
- Logging in data/paper_trading/

Wichtig:
- Minimal-Deployment für die Abgabe, kein produktionsreifes Trading-System.
- yfinance ist nicht garantiert Echtzeit, kann aussetzen.
- Für Intraday ist Alpaca free tier teils delayed -> Daten via yfinance, Orders via Alpaca.
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


# -------------------------
# .env zuverlässig laden (immer aus Projekt-Root)
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


# -------------------------
# Config
# -------------------------
@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"
    gold_symbol: str = "GLD"

    # model choice
    model_name: str = "logreg"  # "logreg" or "rf"

    # trading rules
    entry_threshold: float = 0.55
    exit_threshold: float = 0.50
    max_holding_minutes: int = 15

    # position sizing
    notional_usd: float = 1000.0

    # polling
    poll_seconds: int = 60
    lookback_minutes: int = 180  # >= 15 + rolling windows + safety

    # safety
    trade_only_when_market_open: bool = True

    # paths
    base_dir: Path = PROJECT_ROOT

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "data" / "paper_trading"


# -------------------------
# Alpaca REST client (minimal)
# -------------------------
class AlpacaPaperClient:
    def __init__(self) -> None:
        # EXAKT diese Variablen lesen (env/.env)
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


# -------------------------
# Feature engineering (match scripts/03_data_preparation.py)
# -------------------------
def build_features_from_1m(spy_1m: pd.DataFrame, gld_1m: pd.DataFrame) -> pd.DataFrame:
    """
    spy_1m, gld_1m: 1-min OHLCV DataFrames indexed by UTC timestamps.
    yfinance liefert Spalten: Open/High/Low/Close/Volume.
    """
    spy = spy_1m.copy().rename(columns=str.lower).add_prefix("spy_")
    gld = gld_1m.copy().rename(columns=str.lower).add_prefix("gld_")

    df = spy.join(gld, how="inner").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # SPY
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

    # GLD
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

    # Cross
    df["ret_spy_minus_gld_1m"] = df["spy_ret_1m"] - df["gld_ret_1m"]
    df["ret_spy_minus_gld_15m"] = df["spy_ret_15m"] - df["gld_ret_15m"]
    df["vol_ratio_spy_gld_15m"] = df["spy_roll_std_15m"] / (df["gld_roll_std_15m"] + 1e-12)
    df["price_ratio_spy_gld"] = df["spy_close"] / (df["gld_close"] + 1e-12)

    # Time features
    df["hour"] = df.index.hour
    df["minute_of_day"] = df["hour"] * 60 + df.index.minute
    df["minute_of_day_norm"] = df["minute_of_day"] / (24 * 60)

    return df


def feature_cols_from_train(train_df: pd.DataFrame, target_col: str = "target_up") -> list[str]:
    return [c for c in train_df.columns if c != target_col and not c.startswith("future_ret_")]


# -------------------------
# Model wrapper
# -------------------------
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
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X = feat_row[self.feature_cols]
        if self.cfg.model_name == "logreg":
            assert self.scaler is not None
            Xs = self.scaler.transform(X)
            return float(self.model.predict_proba(Xs)[:, 1][0])
        return float(self.model.predict_proba(X)[:, 1][0])


# -------------------------
# yfinance fetch
# -------------------------
def yf_fetch_1m(symbol: str, lookback_minutes: int) -> pd.DataFrame:
    df = yf.download(symbol, period="1d", interval="1m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.tail(lookback_minutes)

    # yfinance index ist meist tz-aware (US/Eastern) -> UTC
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


# -------------------------
# Main loop
# -------------------------
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

    # local state
    state = {"in_position": False, "entry_time_utc": None}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print(
        f"✅ Paper Trading gestartet | model={cfg.model_name} | "
        f"entry={cfg.entry_threshold} exit={cfg.exit_threshold} | "
        f"alpaca_base={os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')}"
    )

    while True:
        loop_ts = datetime.now(timezone.utc)

        # market open check
        if cfg.trade_only_when_market_open:
            clock = alp.get_clock()
            if not clock.get("is_open", False):
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
        if feat_df.empty:
            print(f"[{loop_ts.isoformat()}] not enough data for rolling features -> sleep")
            time.sleep(cfg.poll_seconds)
            continue

        t = feat_df.index[-1]
        last = feat_df.iloc[[-1]].copy()

        p_up = live_model.predict_p_up(last)
        last_spy_price = float(last["spy_close"].iloc[0])

        # alpaca truth
        pos = alp.get_position(cfg.primary_symbol)
        in_pos_api = pos is not None

        # reconcile local state
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

        # Trading rules
        if not in_pos_api:
            if p_up >= cfg.entry_threshold:
                qty = ensure_int_qty(cfg.notional_usd, last_spy_price)
                if qty > 0:
                    order_resp = alp.submit_market_order(cfg.primary_symbol, "buy", qty)
                    action = f"BUY qty={qty}"
                    state["in_position"] = True
                    state["entry_time_utc"] = loop_ts.isoformat()
        else:
            if (p_up <= cfg.exit_threshold) or (holding_min >= cfg.max_holding_minutes):
                qty = int(float(pos.get("qty", 0))) if pos else 0
                if qty > 0:
                    order_resp = alp.submit_market_order(cfg.primary_symbol, "sell", qty)
                    action = f"SELL qty={qty}"
                    state["in_position"] = False
                    state["entry_time_utc"] = None

        # persist state
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        # log row
        log_row = {
            "timestamp_utc": loop_ts.isoformat(),
            "feature_time_utc": str(t),
            "p_up": p_up,
            "spy_close": last_spy_price,
            "in_position_api": int(in_pos_api),
            "holding_minutes": holding_min,
            "action": action,
        }
        append_log(log_path, log_row)

        print(f"[{loop_ts.isoformat()}] t={t} p_up={p_up:.4f} pos={in_pos_api} hold={holding_min} -> {action}")

        if order_resp is not None:
            with open(orders_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp_utc": loop_ts.isoformat(), "order": order_resp}) + "\n")

        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()