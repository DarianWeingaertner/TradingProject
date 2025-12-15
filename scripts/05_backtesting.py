# scripts/05_backtest.py
"""
Step 05 — Backtesting (Out-of-sample on val.csv)
------------------------------------------------
Ziele:
- Trading-Algorithmus aus trainiertem Modell ableiten
- Entry/Exit Regeln spezifizieren (probability thresholds + max holding)
- Backtest auf VAL (zeitlich später als Train) durchführen
- Performance-Metriken & Plots speichern
- Vergleich zu Buy&Hold (SPY)

Wichtig:
- Keine Data Leakage: Modell wird auf train.csv fitten, Signale/Backtest auf val.csv.
- Execution: Entscheidungen basieren auf Features zur Zeit t; Ausführung am OPEN von t+1.

Outputs:
- data/backtests/trades_{model}.csv
- data/backtests/equity_curve_{model}.csv
- data/backtests/metrics_{model}.json
- figures/backtest_equity_curve_{model}.png
- figures/backtest_drawdown_{model}.png
- figures/backtest_trade_return_hist_{model}.png
- figures/backtest_entry_distribution_hour_{model}.png
- figures/backtest_entries_over_time_{model}.png
- figures/backtest_examples_{model}.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -------------------------
# Config
# -------------------------
@dataclass
class ProjectConfig:
    primary_symbol: str = "SPY"

    prediction_horizon_min: int = 15  # max holding (minutes) in backtest

    # Trading rules
    entry_threshold: float = 0.55
    exit_threshold: float = 0.50

    # Costs (bps = 1/100 of a percent)
    fee_bps: float = 1.0
    slippage_bps: float = 1.0

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
    # index is timestamp string written by pandas -> parse to datetime
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.isna().any():
        raise ValueError(f"Failed to parse timestamps in index for: {path}")

    df = df.sort_index()
    return df


def _feature_columns(df: pd.DataFrame, target_col: str = "target_up") -> list[str]:
    cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if c.startswith("future_ret_"):
            continue
        cols.append(c)
    return cols


def _cost_rate(cfg: ProjectConfig) -> float:
    # total cost per transaction side (entry and exit both apply)
    return (cfg.fee_bps + cfg.slippage_bps) / 10_000.0


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


# -------------------------
# Backtest Engine
# -------------------------
class Backtester:
    def __init__(self, cfg: ProjectConfig, model_name: str = "logreg") -> None:
        self.cfg = cfg
        self.model_name = model_name.lower().strip()
        if self.model_name not in {"logreg", "rf"}:
            raise ValueError("model_name must be one of: logreg, rf")

        self.scaler: StandardScaler | None = None
        self.model = None

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

    def predict_proba(self, df: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
        X = df[feat_cols]
        if X.isnull().any().any():
            raise ValueError("NaNs in backtest features. Ensure val.csv is clean.")

        if self.model_name == "logreg":
            assert self.scaler is not None and self.model is not None
            Xs = self.scaler.transform(X)
            p = self.model.predict_proba(Xs)[:, 1]
        else:
            assert self.model is not None
            p = self.model.predict_proba(X)[:, 1]

        return pd.Series(p, index=df.index, name="p_up")

    # ---- trading rules ----
    def _should_enter(self, p_up: float) -> bool:
        return p_up >= self.cfg.entry_threshold

    def _should_exit(self, p_up: float, minutes_in_trade: int) -> bool:
        if minutes_in_trade >= self.cfg.prediction_horizon_min:
            return True
        return p_up <= self.cfg.exit_threshold

    # ---- backtest ----
    def run_backtest(self, val_df: pd.DataFrame, p_up: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Long-only, fully invested when in position.
        Decisions at time t, execution at OPEN of t+1.
        Mark-to-market uses SPY close at each timestamp.
        """
        if "spy_open" not in val_df.columns or "spy_close" not in val_df.columns:
            raise ValueError("val_df must contain spy_open and spy_close columns (from Step 03).")

        cost = _cost_rate(self.cfg)

        idx = val_df.index
        if len(idx) < 3:
            raise ValueError("val_df too short for backtest.")

        # state
        cash = 1.0
        shares = 0.0
        in_pos = False
        entry_i: int | None = None
        entry_time: pd.Timestamp | None = None
        entry_price_eff: float | None = None

        # outputs
        equity_records = []
        trades = []

        # For example plots: track entries/exits
        entry_marks = []
        exit_marks = []

        # iterate over bars; use i and i+1 (execution on next bar open)
        for i in range(len(idx) - 1):
            t = idx[i]
            t_next = idx[i + 1]

            close_t = float(val_df.loc[t, "spy_close"])
            open_next = float(val_df.loc[t_next, "spy_open"])
            p_t = float(p_up.loc[t])

            # mark-to-market equity at time t (using close)
            if in_pos:
                equity_t = shares * close_t
            else:
                equity_t = cash

            equity_records.append({"timestamp": t, "equity": equity_t, "in_position": int(in_pos), "p_up": p_t})

            # decide action (executed at t_next open)
            if not in_pos:
                if self._should_enter(p_t):
                    # enter at next open with costs
                    fill = open_next * (1.0 + cost)
                    if fill <= 0:
                        continue
                    shares = cash / fill
                    cash = 0.0
                    in_pos = True

                    entry_i = i + 1
                    entry_time = t_next
                    entry_price_eff = fill

                    entry_marks.append((t_next, open_next))
            else:
                assert entry_i is not None and entry_time is not None and entry_price_eff is not None
                minutes_in_trade = (i + 1) - entry_i  # minutes elapsed until next open
                if self._should_exit(p_t, minutes_in_trade):
                    # exit at next open with costs
                    exit_fill = open_next * (1.0 - cost)
                    cash = shares * exit_fill
                    shares = 0.0
                    in_pos = False

                    # trade stats
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
                    exit_marks.append((t_next, open_next))

                    entry_i = None
                    entry_time = None
                    entry_price_eff = None

        # final mark-to-market / liquidation on last close
        t_last = idx[-1]
        close_last = float(val_df.loc[t_last, "spy_close"])
        p_last = float(p_up.loc[t_last])

        if in_pos:
            # liquidate at last close with costs (approx)
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
            exit_marks.append((t_last, close_last))

        equity_records.append(
            {"timestamp": t_last, "equity": cash, "in_position": 0, "p_up": p_last}
        )

        equity_df = pd.DataFrame(equity_records).set_index("timestamp").sort_index()
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
            trades_df = trades_df.sort_values("entry_time")

        metrics = self._compute_metrics(equity_df, trades_df, val_df)

        # store marks for plotting
        metrics["_entry_marks_count"] = int(len(entry_marks))
        metrics["_exit_marks_count"] = int(len(exit_marks))

        return equity_df, trades_df, metrics

    def _compute_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        eq = equity_df["equity"].astype(float)
        ret_series = eq.pct_change().fillna(0.0)

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

        # drawdown
        running_max = eq.cummax()
        dd = (eq / running_max) - 1.0
        max_dd = float(dd.min())

        # simple “minute Sharpe” (not annualized; optional)
        mu = float(ret_series.mean())
        sig = float(ret_series.std(ddof=0))
        sharpe_min = _safe_div(mu, sig) if sig > 0 else float("nan")

        n_trades = int(len(trades_df))
        win_rate = float((trades_df["return"] > 0).mean()) if n_trades > 0 else float("nan")
        avg_trade = float(trades_df["return"].mean()) if n_trades > 0 else float("nan")
        med_trade = float(trades_df["return"].median()) if n_trades > 0 else float("nan")

        # buy&hold on SPY for same VAL period
        spy_close = val_df["spy_close"].astype(float)
        buy_hold_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1.0)

        return {
            "model": self.model_name,
            "entry_threshold": self.cfg.entry_threshold,
            "exit_threshold": self.cfg.exit_threshold,
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
# Plotting
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
    # clip tails for readability
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
        # create empty plots for consistency
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

    # by hour
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

    # over time (per day)
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
    """
    Simple example plot: last N days of VAL with entry/exit vertical lines.
    """
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
        # only mark trades that overlap the window
        for _, tr in trades_df.iterrows():
            et = pd.to_datetime(tr["entry_time"], utc=True)
            xt = pd.to_datetime(tr["exit_time"], utc=True)
            if et >= window.index.min() and et <= window.index.max():
                plt.axvline(et, linewidth=1)
            if xt >= window.index.min() and xt <= window.index.max():
                plt.axvline(xt, linewidth=1)

    plt.title(f"Backtest Examples (last {n_days} days of VAL) ({tag})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("SPY close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--entry", type=float, default=0.55)
    parser.add_argument("--exit", type=float, default=0.50)
    parser.add_argument("--hold", type=int, default=15)
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--slippage_bps", type=float, default=1.0)
    args = parser.parse_args()

    cfg = ProjectConfig(
        entry_threshold=args.entry,
        exit_threshold=args.exit,
        prediction_horizon_min=args.hold,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    train_path = cfg.processed_data_dir / "train.csv"
    val_path = cfg.processed_data_dir / "val.csv"

    train_df = _load_processed_csv(train_path)
    val_df = _load_processed_csv(val_path)

    bt = Backtester(cfg, model_name=args.model)
    feat_cols = bt.fit_model(train_df)
    p_up = bt.predict_proba(val_df, feat_cols=feat_cols)

    equity_df, trades_df, metrics = bt.run_backtest(val_df, p_up)

    # save outputs
    cfg.backtests_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model

    trades_out = cfg.backtests_dir / f"trades_{tag}.csv"
    equity_out = cfg.backtests_dir / f"equity_curve_{tag}.csv"
    metrics_out = cfg.backtests_dir / f"metrics_{tag}.json"

    trades_df.to_csv(trades_out, index=False)
    equity_df.to_csv(equity_out)

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # plots
    p1 = plot_equity_vs_buyhold(cfg, equity_df, val_df, tag=tag)
    p2 = plot_drawdown(cfg, equity_df, tag=tag)
    p3 = plot_trade_return_hist(cfg, trades_df, tag=tag)
    p4, p5 = plot_entry_distribution(cfg, trades_df, tag=tag)
    p6 = plot_examples(cfg, val_df, trades_df, tag=tag, n_days=3)

    # console summary
    print("\n=== Backtest Summary (VAL, out-of-sample) ===")
    for k in [
        "model",
        "val_start",
        "val_end",
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
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")
    print(f" - {p4}")
    print(f" - {p5}")
    print(f" - {p6}")


if __name__ == "__main__":
    main()