# scripts/05_backtesting.py
"""
Step 05 — Backtesting (Out-of-sample on val.csv) with Tune/Test split + optional calibration + threshold sweep
------------------------------------------------------------------------------------------------------------
Ziele:
- Modell auf train.csv fitten (kein Leakage)
- VAL zeitlich splitten: Tune (für Calibration + Threshold Auswahl) und Test (echtes Report)
- Optional: Probability Calibration (sigmoid/isotonic) nur auf Tune
- Optional: Threshold Sweep auf Tune, Auswahl nach Metric, Report auf Test
- Trading Regeln identisch zu Paper Trading:
  - Decision auf Bar t (features/p_up[t])
  - Execution am OPEN von t+1
  - Entry: p_up >= entry_threshold (mit No-Trade Zone via Hysterese möglich)
  - Exit:  p_up <= exit_threshold ODER max_holding_minutes erreicht
- Kosten: fee_bps + slippage_bps pro Side

Outputs:
- data/backtests/trades_{tag}.csv
- data/backtests/equity_curve_{tag}.csv
- data/backtests/metrics_{tag}.json
- data/backtests/sweep_results_{tag}.csv (wenn --sweep)
- figures/... (wie gehabt) + zusätzlich p_up quantile plot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


# -------------------------
# Config
# -------------------------
@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"

    prediction_horizon_min: int = 15  # max holding (minutes)

    # Base thresholds (can be overridden by sweep)
    entry_threshold: float = 0.55
    exit_threshold: float = 0.50

    # Optional extra hysteresis: effective_entry=max(entry, 0.5+margin), effective_exit=min(exit, 0.5-margin)
    no_trade_margin: float = 0.00

    # Costs (bps)
    fee_bps: float = 1.0
    slippage_bps: float = 1.0

    # Val split: first part is Tune, second is Test
    tune_frac: float = 0.50

    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def backtests_dir(self) -> Path:
        return self.base_dir / "data" / "backtests"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"


# -------------------------
# Utilities
# -------------------------
def _load_processed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.isna().any():
        raise ValueError(f"Failed to parse timestamps in index for: {path}")
    return df.sort_index()


def _feature_columns(df: pd.DataFrame, target_col: str = "target_up") -> list[str]:
    return [c for c in df.columns if c != target_col and not c.startswith("future_ret_")]


def _cost_rate(cfg: ProjectConfig) -> float:
    return (cfg.fee_bps + cfg.slippage_bps) / 10_000.0


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _time_split(df: pd.DataFrame, tune_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.1 <= tune_frac <= 0.9):
        raise ValueError("tune_frac should be between 0.1 and 0.9 (reasonable split).")
    n = len(df)
    if n < 1000:
        raise ValueError("VAL too short for a meaningful tune/test split.")
    cut = int(n * tune_frac)
    tune = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return tune, test


# -------------------------
# Backtest Engine
# -------------------------
class Backtester:
    def __init__(self, cfg: ProjectConfig, model_name: str = "logreg", calibrate: str = "none") -> None:
        self.cfg = cfg
        self.model_name = model_name.lower().strip()
        if self.model_name not in {"logreg", "rf"}:
            raise ValueError("model_name must be one of: logreg, rf")

        self.calibrate = calibrate.lower().strip()
        if self.calibrate not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("calibrate must be one of: none, sigmoid, isotonic")

        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.calibrator = None  # CalibratedClassifierCV with cv='prefit'

    # ---- training ----
    def fit_model(self, train_df: pd.DataFrame) -> list[str]:
        target_col = "target_up"
        feat_cols = _feature_columns(train_df, target_col=target_col)

        X_train = train_df[feat_cols]
        y_train = train_df[target_col].astype(int)

        if X_train.isnull().any().any():
            raise ValueError("NaNs in train features. Re-run Step 03 and ensure dropna().")

        if self.model_name == "logreg":
            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X_train)
            self.model = LogisticRegression(max_iter=800)
            self.model.fit(Xs, y_train)
        else:
            self.scaler = None
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)

        return feat_cols

    def fit_calibrator_on_tune(self, tune_df: pd.DataFrame, feat_cols: list[str]) -> None:
        if self.calibrate == "none":
            self.calibrator = None
            return
        if self.model is None:
            raise RuntimeError("Base model not fitted.")

        X_tune = tune_df[feat_cols]
        y_tune = tune_df["target_up"].astype(int)

        if X_tune.isnull().any().any():
            raise ValueError("NaNs in tune features. Ensure val.csv is clean.")

        if self.model_name == "logreg":
            assert self.scaler is not None
            X_tune_in = self.scaler.transform(X_tune)
        else:
            X_tune_in = X_tune

        # Calibrate prefit model on Tune only (no leakage into Test)
        self.calibrator = CalibratedClassifierCV(self.model, method=self.calibrate, cv="prefit")
        self.calibrator.fit(X_tune_in, y_tune)

    def predict_proba(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
        X = df[feat_cols]
        if X.isnull().any().any():
            raise ValueError("NaNs in backtest features. Ensure val.csv is clean.")

        if self.model_name == "logreg":
            assert self.scaler is not None and self.model is not None
            X_in = self.scaler.transform(X)
        else:
            assert self.model is not None
            X_in = X

        if self.calibrator is not None:
            p = self.calibrator.predict_proba(X_in)[:, 1]
        else:
            p = self.model.predict_proba(X_in)[:, 1]

        return pd.Series(p, index=df.index, name="p_up")

    # ---- trading rules ----
    def _effective_thresholds(self) -> tuple[float, float]:
        entry = float(self.cfg.entry_threshold)
        exit_ = float(self.cfg.exit_threshold)

        m = float(self.cfg.no_trade_margin)
        if m > 0:
            entry = max(entry, 0.5 + m)
            exit_ = min(exit_, 0.5 - m)

        # sanity: must be entry > exit for hysteresis; otherwise you churn
        if entry <= exit_:
            # force a tiny hysteresis instead of silently running nonsense
            entry = max(entry, exit_ + 1e-6)
        return entry, exit_

    def _should_enter(self, p_up: float) -> bool:
        entry, _ = self._effective_thresholds()
        return p_up >= entry

    def _should_exit(self, p_up: float, minutes_in_trade: int) -> bool:
        _, exit_ = self._effective_thresholds()
        if minutes_in_trade >= self.cfg.prediction_horizon_min:
            return True
        return p_up <= exit_

    # ---- backtest ----
    def run_backtest(self, val_df: pd.DataFrame, p_up: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if "spy_open" not in val_df.columns or "spy_close" not in val_df.columns:
            raise ValueError("val_df must contain spy_open and spy_close columns (from Step 03).")

        cost = _cost_rate(self.cfg)
        idx = val_df.index
        if len(idx) < 3:
            raise ValueError("val_df too short for backtest.")

        cash = 1.0
        shares = 0.0
        in_pos = False

        entry_i: int | None = None
        entry_time: pd.Timestamp | None = None
        entry_price_eff: float | None = None

        equity_records = []
        trades = []

        for i in range(len(idx) - 1):
            t = idx[i]
            t_next = idx[i + 1]

            close_t = float(val_df.loc[t, "spy_close"])
            open_next = float(val_df.loc[t_next, "spy_open"])
            p_t = float(p_up.loc[t])

            equity_t = shares * close_t if in_pos else cash
            equity_records.append({"timestamp": t, "equity": equity_t, "in_position": int(in_pos), "p_up": p_t})

            if not in_pos:
                if self._should_enter(p_t):
                    fill = open_next * (1.0 + cost)
                    if fill <= 0:
                        continue
                    shares = cash / fill
                    cash = 0.0
                    in_pos = True

                    entry_i = i + 1
                    entry_time = t_next
                    entry_price_eff = fill
            else:
                assert entry_i is not None and entry_time is not None and entry_price_eff is not None
                minutes_in_trade = (i + 1) - entry_i
                if self._should_exit(p_t, minutes_in_trade):
                    exit_fill = open_next * (1.0 - cost)
                    cash = shares * exit_fill
                    shares = 0.0
                    in_pos = False

                    gross_ret = (exit_fill / entry_price_eff) - 1.0
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": t_next,
                            "entry_price_eff": entry_price_eff,
                            "exit_price_eff": exit_fill,
                            "hold_minutes": minutes_in_trade,
                            "return": gross_ret,
                        }
                    )

                    entry_i = None
                    entry_time = None
                    entry_price_eff = None

        # liquidation on last close
        t_last = idx[-1]
        close_last = float(val_df.loc[t_last, "spy_close"])
        p_last = float(p_up.loc[t_last])

        if in_pos:
            exit_fill = close_last * (1.0 - cost)
            cash = shares * exit_fill
            shares = 0.0
            in_pos = False

            assert entry_time is not None and entry_price_eff is not None and entry_i is not None
            hold_minutes = (len(idx) - 1) - entry_i
            gross_ret = (exit_fill / entry_price_eff) - 1.0
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": t_last,
                    "entry_price_eff": entry_price_eff,
                    "exit_price_eff": exit_fill,
                    "hold_minutes": hold_minutes,
                    "return": gross_ret,
                }
            )

        equity_records.append({"timestamp": t_last, "equity": cash, "in_position": 0, "p_up": p_last})

        equity_df = pd.DataFrame(equity_records).set_index("timestamp").sort_index()
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
            trades_df = trades_df.sort_values("entry_time")

        metrics = self._compute_metrics(equity_df, trades_df, val_df)
        return equity_df, trades_df, metrics

    def _compute_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        eq = equity_df["equity"].astype(float)
        ret_series = eq.pct_change().fillna(0.0)

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

        running_max = eq.cummax()
        dd = (eq / running_max) - 1.0
        max_dd = float(dd.min())

        mu = float(ret_series.mean())
        sig = float(ret_series.std(ddof=0))
        sharpe_min = _safe_div(mu, sig) if sig > 0 else float("nan")

        n_trades = int(len(trades_df))
        win_rate = float((trades_df["return"] > 0).mean()) if n_trades > 0 else float("nan")
        avg_trade = float(trades_df["return"].mean()) if n_trades > 0 else float("nan")
        med_trade = float(trades_df["return"].median()) if n_trades > 0 else float("nan")

        spy_close = val_df["spy_close"].astype(float)
        buy_hold_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1.0)

        entry_eff, exit_eff = self._effective_thresholds()

        return {
            "model": self.model_name,
            "calibrate": self.calibrate,
            "entry_threshold_base": self.cfg.entry_threshold,
            "exit_threshold_base": self.cfg.exit_threshold,
            "no_trade_margin": self.cfg.no_trade_margin,
            "entry_threshold_effective": entry_eff,
            "exit_threshold_effective": exit_eff,
            "max_holding_minutes": self.cfg.prediction_horizon_min,
            "fee_bps": self.cfg.fee_bps,
            "slippage_bps": self.cfg.slippage_bps,
            "val_start": str(equity_df.index.min()),
            "val_end": str(equity_df.index.max()),
            "total_return": total_return,
            "buy_hold_return": buy_hold_return,
            "max_drawdown": max_dd,
            "sharpe_minute": sharpe_min,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade,
            "median_trade_return": med_trade,
        }


# -------------------------
# Diagnostics: p_up quantiles vs future_ret_15m
# -------------------------
def plot_pup_quantiles(cfg: ProjectConfig, df: pd.DataFrame, p_up: pd.Series, tag: str) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.figures_dir / f"pup_quantiles_{tag}.png"

    if "future_ret_15m" not in df.columns:
        # If you don't have it in val.csv, you can't do this diagnostic.
        plt.figure(figsize=(10, 4))
        plt.title(f"p_up quantiles ({tag}) — future_ret_15m missing")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    tmp = df.copy()
    tmp["p_up"] = p_up.astype(float)

    # 10 quantiles
    tmp = tmp.dropna(subset=["p_up", "future_ret_15m"])
    if len(tmp) < 1000:
        plt.figure(figsize=(10, 4))
        plt.title(f"p_up quantiles ({tag}) — insufficient data")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    tmp["q"] = pd.qcut(tmp["p_up"], 10, labels=False, duplicates="drop")
    grp = tmp.groupby("q")["future_ret_15m"].mean()

    plt.figure(figsize=(10, 4))
    plt.bar(grp.index.astype(int), grp.values)
    plt.title(f"Mean future_ret_15m by p_up decile ({tag})")
    plt.xlabel("p_up decile (low→high)")
    plt.ylabel("Mean future_ret_15m")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# -------------------------
# Plotting (existing)
# -------------------------
def plot_equity_vs_buyhold(cfg: ProjectConfig, equity_df: pd.DataFrame, val_df: pd.DataFrame, tag: str) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    eq = equity_df["equity"].astype(float)
    spy = val_df["spy_close"].astype(float)
    buyhold = spy / spy.iloc[0] * eq.iloc[0]

    plt.figure(figsize=(12, 5))
    plt.plot(eq.index, eq.values, label="Strategy Equity")
    plt.plot(buyhold.index, buyhold.values, label="Buy&Hold SPY")
    plt.title(f"Backtest Equity Curve ({tag})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    plt.tight_layout()

    out = cfg.figures_dir / f"backtest_equity_curve_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_drawdown(cfg: ProjectConfig, equity_df: pd.DataFrame, tag: str) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    eq = equity_df["equity"].astype(float)
    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0

    plt.figure(figsize=(12, 4))
    plt.plot(dd.index, dd.values)
    plt.title(f"Backtest Drawdown ({tag})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Drawdown")
    plt.tight_layout()

    out = cfg.figures_dir / f"backtest_drawdown_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_trade_return_hist(cfg: ProjectConfig, trades_df: pd.DataFrame, tag: str) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.figures_dir / f"backtest_trade_return_hist_{tag}.png"

    plt.figure(figsize=(8, 5))
    if trades_df.empty:
        plt.title(f"Trade Return Histogram ({tag}) — no trades")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    rets = trades_df["return"].astype(float)
    lo, hi = rets.quantile(0.01), rets.quantile(0.99)
    rets = rets.clip(lo, hi)

    plt.hist(rets.values, bins=40)
    plt.title(f"Trade Return Histogram ({tag})")
    plt.xlabel("Trade return")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_entry_distribution(cfg: ProjectConfig, trades_df: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    out_hour = cfg.figures_dir / f"backtest_entry_distribution_hour_{tag}.png"
    out_time = cfg.figures_dir / f"backtest_entries_over_time_{tag}.png"

    if trades_df.empty:
        plt.figure(figsize=(8, 4))
        plt.title(f"Entry distribution by hour ({tag}) — no trades")
        plt.tight_layout()
        plt.savefig(out_hour, dpi=150)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.title(f"Entries over time ({tag}) — no trades")
        plt.tight_layout()
        plt.savefig(out_time, dpi=150)
        plt.close()
        return out_hour, out_time

    entries = pd.to_datetime(trades_df["entry_time"], utc=True)

    hours = entries.dt.hour
    counts = hours.value_counts().sort_index()

    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(int), counts.values)
    plt.title(f"Entry Distribution by Hour (UTC) ({tag})")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Entries")
    plt.tight_layout()
    plt.savefig(out_hour, dpi=150)
    plt.close()

    per_day = entries.dt.floor("D").value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.plot(per_day.index, per_day.values)
    plt.title(f"Entries over Time (per day) ({tag})")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Entries")
    plt.tight_layout()
    plt.savefig(out_time, dpi=150)
    plt.close()

    return out_hour, out_time


def plot_examples(cfg: ProjectConfig, val_df: pd.DataFrame, trades_df: pd.DataFrame, tag: str, n_days: int = 3) -> Path:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.figures_dir / f"backtest_examples_{tag}.png"

    spy_close = val_df["spy_close"].astype(float)
    if len(spy_close) == 0:
        plt.figure(figsize=(12, 5))
        plt.title(f"Examples ({tag}) — no data")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    end = spy_close.index.max()
    start = end - pd.Timedelta(days=n_days)
    window = spy_close.loc[start:end]

    plt.figure(figsize=(12, 5))
    plt.plot(window.index, window.values, label="SPY close")

    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            et = pd.to_datetime(tr["entry_time"], utc=True)
            xt = pd.to_datetime(tr["exit_time"], utc=True)
            if et >= window.index.min() and et <= window.index.max():
                plt.axvline(et, linewidth=1)
            if xt >= window.index.min() and xt <= window.index.max():
                plt.axvline(xt, linewidth=1)

    plt.title(f"Backtest Examples (last {n_days} days of TEST) ({tag})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("SPY close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# -------------------------
# Sweep
# -------------------------
def _frange(a: float, b: float, step: float) -> list[float]:
    vals = []
    x = a
    # inclusive-ish
    while x <= b + 1e-12:
        vals.append(round(float(x), 6))
        x += step
    return vals


def run_threshold_sweep(
    cfg: ProjectConfig,
    bt: Backtester,
    tune_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    metric: str = "total_return",
    entry_grid: tuple[float, float, float] = (0.52, 0.70, 0.02),
    exit_grid: tuple[float, float, float] = (0.30, 0.52, 0.02),
    hold_grid: tuple[int, int, int] = (5, 60, 5),
) -> tuple[pd.DataFrame, dict]:
    metric = metric.strip().lower()
    if metric not in {"total_return", "sharpe_minute", "max_drawdown"}:
        raise ValueError("metric must be one of: total_return, sharpe_minute, max_drawdown")

    p_tune = bt.predict_proba(tune_df, feat_cols)
    p_test = bt.predict_proba(test_df, feat_cols)

    entries = _frange(*entry_grid)
    exits = _frange(*exit_grid)
    holds = list(range(hold_grid[0], hold_grid[1] + 1, hold_grid[2]))

    rows = []
    best = None
    best_score = None

    for e in entries:
        for x in exits:
            if x >= e:
                continue
            for h in holds:
                cfg_tmp = ProjectConfig(
                    entry_threshold=e,
                    exit_threshold=x,
                    prediction_horizon_min=h,
                    fee_bps=cfg.fee_bps,
                    slippage_bps=cfg.slippage_bps,
                    no_trade_margin=cfg.no_trade_margin,
                    tune_frac=cfg.tune_frac,
                )
                bt_tmp = Backtester(cfg_tmp, model_name=bt.model_name, calibrate=bt.calibrate)
                # reuse already fitted model/scaler/calibrator
                bt_tmp.model = bt.model
                bt_tmp.scaler = bt.scaler
                bt_tmp.calibrator = bt.calibrator

                _, _, m_tune = bt_tmp.run_backtest(tune_df, p_tune)
                _, _, m_test = bt_tmp.run_backtest(test_df, p_test)

                row = {
                    "entry": e,
                    "exit": x,
                    "hold": h,
                    "tune_total_return": m_tune["total_return"],
                    "tune_sharpe_minute": m_tune["sharpe_minute"],
                    "tune_max_drawdown": m_tune["max_drawdown"],
                    "test_total_return": m_test["total_return"],
                    "test_sharpe_minute": m_test["sharpe_minute"],
                    "test_max_drawdown": m_test["max_drawdown"],
                    "test_n_trades": m_test["n_trades"],
                }
                rows.append(row)

                score = row[f"tune_{metric}"]
                # for max_drawdown, "higher" is better (less negative) -> maximize it
                if best_score is None or (score > best_score):
                    best_score = score
                    best = row

    res = pd.DataFrame(rows).sort_values(by=f"tune_{metric}", ascending=False)
    assert best is not None
    return res, best


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])

    # base params (used when not sweeping)
    parser.add_argument("--entry", type=float, default=0.55)
    parser.add_argument("--exit", type=float, default=0.50)
    parser.add_argument("--hold", type=int, default=15)

    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--slippage_bps", type=float, default=1.0)

    parser.add_argument("--tune_frac", type=float, default=0.50)
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--no_trade_margin", type=float, default=0.00)

    # sweep
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_metric", type=str, default="sharpe_minute", choices=["total_return", "sharpe_minute", "max_drawdown"])

    parser.add_argument("--sweep_entry_min", type=float, default=0.52)
    parser.add_argument("--sweep_entry_max", type=float, default=0.70)
    parser.add_argument("--sweep_entry_step", type=float, default=0.02)

    parser.add_argument("--sweep_exit_min", type=float, default=0.30)
    parser.add_argument("--sweep_exit_max", type=float, default=0.52)
    parser.add_argument("--sweep_exit_step", type=float, default=0.02)

    parser.add_argument("--sweep_hold_min", type=int, default=5)
    parser.add_argument("--sweep_hold_max", type=int, default=60)
    parser.add_argument("--sweep_hold_step", type=int, default=5)

    args = parser.parse_args()

    cfg = ProjectConfig(
        entry_threshold=args.entry,
        exit_threshold=args.exit,
        prediction_horizon_min=args.hold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        tune_frac=args.tune_frac,
        calibrate=getattr(args, "calibrate", "none") if False else args.calibrate,  # keep linters calm
        no_trade_margin=args.no_trade_margin,
    )

    train_path = cfg.processed_data_dir / "train.csv"
    val_path = cfg.processed_data_dir / "val.csv"

    train_df = _load_processed_csv(train_path)
    val_df = _load_processed_csv(val_path)

    tune_df, test_df = _time_split(val_df, cfg.tune_frac)

    bt = Backtester(cfg, model_name=args.model, calibrate=args.calibrate)
    feat_cols = bt.fit_model(train_df)

    # Fit calibrator on Tune ONLY (optional)
    bt.fit_calibrator_on_tune(tune_df, feat_cols)

    # Either sweep or run single
    if args.sweep:
        sweep_df, best = run_threshold_sweep(
            cfg=cfg,
            bt=bt,
            tune_df=tune_df,
            test_df=test_df,
            feat_cols=feat_cols,
            metric=args.sweep_metric,
            entry_grid=(args.sweep_entry_min, args.sweep_entry_max, args.sweep_entry_step),
            exit_grid=(args.sweep_exit_min, args.sweep_exit_max, args.sweep_exit_step),
            hold_grid=(args.sweep_hold_min, args.sweep_hold_max, args.sweep_hold_step),
        )

        # set cfg to best for final test report
        cfg.entry_threshold = float(best["entry"])
        cfg.exit_threshold = float(best["exit"])
        cfg.prediction_horizon_min = int(best["hold"])

    # Predict on TEST and run backtest
    p_test = bt.predict_proba(test_df, feat_cols=feat_cols)
    equity_df, trades_df, metrics = bt.run_backtest(test_df, p_test)

    # Tag
    tag = f"{args.model}_cal{args.calibrate}_m{cfg.no_trade_margin:.2f}_t{cfg.tune_frac:.2f}_e{cfg.entry_threshold:.2f}_x{cfg.exit_threshold:.2f}_h{cfg.prediction_horizon_min}"
    cfg.backtests_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    trades_out = cfg.backtests_dir / f"trades_{tag}.csv"
    equity_out = cfg.backtests_dir / f"equity_curve_{tag}.csv"
    metrics_out = cfg.backtests_dir / f"metrics_{tag}.json"

    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots on TEST
    p1 = plot_equity_vs_buyhold(cfg, equity_df, test_df, tag=tag)
    p2 = plot_drawdown(cfg, equity_df, tag=tag)
    p3 = plot_trade_return_hist(cfg, trades_df, tag=tag)
    p4, p5 = plot_entry_distribution(cfg, trades_df, tag=tag)
    p6 = plot_examples(cfg, test_df, trades_df, tag=tag, n_days=3)
    p7 = plot_pup_quantiles(cfg, test_df, p_test, tag=tag)

    sweep_out = None
    if args.sweep:
        sweep_out = cfg.backtests_dir / f"sweep_results_{tag}.csv"
        sweep_df.to_csv(sweep_out, index=False)

    # Console summary
    print("\n=== Backtest Summary (TEST, out-of-sample) ===")
    for k in [
        "model",
        "calibrate",
        "val_start",
        "val_end",
        "entry_threshold_effective",
        "exit_threshold_effective",
        "total_return",
        "buy_hold_return",
        "max_drawdown",
        "sharpe_minute",
        "n_trades",
        "win_rate",
        "avg_trade_return",
        "median_trade_return",
    ]:
        print(f"{k}: {metrics.get(k)}")

    print("\nSaved:")
    print(f" - {trades_out}")
    print(f" - {equity_out}")
    print(f" - {metrics_out}")
    if sweep_out is not None:
        print(f" - {sweep_out}")
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")
    print(f" - {p4}")
    print(f" - {p5}")
    print(f" - {p6}")
    print(f" - {p7}")


if __name__ == "__main__":
    main()