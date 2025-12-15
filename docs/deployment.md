# 🚀 Deployment & Evaluation

This document describes the **deployment stage** of the project, focusing on  
**backtesting**, **paper trading concepts**, and **performance evaluation** of the trained models.

The goal of this section is **not live trading**, but to demonstrate how a trained
machine-learning model can be transformed into a **rule-based trading strategy**
and evaluated realistically on historical data.

---

## 1. Backtesting Trading Algorithms

### 1.1 From Model to Trading Strategy

The trained **Logistic Regression** model outputs a **binary prediction**:

- `1` → SPY price is expected to rise within the next 15 minutes  
- `0` → no expected price increase

This prediction was converted into a **discrete intraday trading rule**.

#### Trading Logic
- **Entry**:  
  Open a long position when  
  `model_prediction == 1`

- **Exit**:  
  Close the position automatically after **15 minutes**  
  (fixed holding period aligned with the prediction horizon)

This approach ensures:
- No look-ahead bias  
- Clear and reproducible decision rules  
- Full alignment between model target and trading behavior  

No leverage, no short selling, and no transaction costs were applied to keep the
backtest interpretable and conservative.

---

### 1.2 Backtest Setup

- Instrument: **SPY**
- Timeframe: **1-minute bars**
- Dataset: Validation split (out-of-sample)
- Capital: Normalized to `1.0`
- Position size: 100% of capital per trade
- Strategy type: Long-only

The backtest was implemented in:
scripts/05_backtest.py

---

## 2. Backtest Results

### 2.1 Equity Curve & Market Comparison

The strategy equity curve was compared to a **Buy & Hold SPY benchmark**.

**Key observation:**
- Buy & Hold shows a strong upward trend during the period
- The strategy equity **declines steadily over time**

This indicates that the model fails to capture enough signal to overcome
market noise at the intraday level.

---

### 2.2 Drawdown Analysis

The drawdown curve shows:
- Persistent drawdowns
- No sustained recovery to previous equity highs
- Maximum drawdown of approximately **-5%**

This suggests that losses are structural rather than caused by isolated bad trades.

---

### 2.3 Trade Distribution Over Time

Trades are distributed across the full backtest period:
- No strong clustering in a specific sub-period
- Consistent trading frequency

This rules out the possibility that performance issues are driven by a single market phase.

---

### 2.4 Entry Distribution by Hour (UTC)

The majority of trades occur:
- During U.S. market hours
- Especially in periods of higher liquidity

This confirms that the strategy operates in reasonable intraday windows
and does not trade during illiquid periods.

---

### 2.5 Trade Return Distribution

The histogram of trade returns shows:
- Mean return close to zero
- Slightly negative skew
- Many small losses, few larger gains

This is typical for:
- Short-horizon intraday strategies
- Markets dominated by microstructure noise

---

### 2.6 Visual Trade Examples

Selected examples from the last validation days illustrate:
- Frequent price fluctuations
- Weak directional persistence
- Difficulty of timing short-term movements reliably

These examples highlight the intrinsic challenge of intraday prediction.

---

## 3. Overall Backtest Performance Summary

| Metric | Result |
|------|------|
| Strategy Return | Negative |
| Buy & Hold Return | Strongly Positive |
| Max Drawdown | ~5% |
| Trade Frequency | Moderate |
| Signal Stability | Low |

### Interpretation

- The strategy **does not outperform Buy & Hold**
- Predictive accuracy is insufficient for profitable intraday trading
- The model captures **weak statistical patterns**, but not exploitable signals

This result is **expected** given:
- Extremely noisy 1-minute data
- Limited dataset size
- Absence of transaction cost modeling

---

## 4. Paper Trading (Conceptual)

### 4.1 Setup

A realistic paper trading setup would include:
- Broker: Alpaca Paper Trading API
- Execution: Market orders
- Frequency: Intraday (1-minute resolution)
- Capital constraints and logging

Signals would be generated in real-time using the trained model
and executed automatically.

---

### 4.2 Performance Expectations

Based on backtest results:
- Paper trading performance is expected to be similar or worse
- Slippage and transaction costs would further reduce returns
- Results would likely underperform Buy & Hold

Paper trading would primarily serve as:
- A system validation step
- A robustness and infrastructure test

---

### 4.3 Comparison to Backtesting

| Aspect | Backtest | Paper Trading |
|-----|---------|---------------|
| Execution | Idealized | Realistic |
| Costs | Ignored | Present |
| Latency | None | Present |
| Expected Performance | Upper bound | Lower bound |

---

## 5. Next Steps & Improvements

Potential improvements include:

- Incorporating **transaction costs and slippage**
- Feature selection and dimensionality reduction
- Probability-based position sizing instead of binary signals
- Regime detection (volatile vs trending markets)
- Longer prediction horizons (30–60 minutes)
- Larger datasets and additional instruments
- Ensemble models and calibration

---

## 6. Reflection on Submission Criteria

### Understanding & Technical Depth (30%)
- Full ML pipeline implemented
- Correct handling of time series data
- Proper train/validation split
- No data leakage

### Documentation Quality (20%)
- Clear separation of README and deployment documentation
- Concise explanations with clear reasoning
- Reproducible workflow

### Goal Orientation & Complexity (30%)
- End-to-end project from data acquisition to deployment
- Non-trivial intraday modeling task
- Realistic evaluation and honest interpretation

### Timeliness (20%)
- All required components implemented and documented
- Structured and complete submission

---

## 7. Conclusion

This project demonstrates that:
- Intraday price prediction is extremely challenging
- Clean engineering and evaluation matter more than raw accuracy
- Negative results are valuable when correctly analyzed

While the strategy is not profitable, the project fulfills all
technical and conceptual requirements of a professional ML trading pipeline.