# Experiment 1.1 — Problem Definition & Data Acquisition

## 1. Problem Definition

Wir möchten ein Modell entwickeln, das vorhersagt, ob der MSCI World ETF (URTH) **bis zum Ende des aktuellen Handelstags** steigt oder nicht.  
Dazu nutzen wir **Intraday-Daten** (z. B. minuten- oder stundenbasierte Kursdaten) anstatt nur einen End-of-Day-Wert pro Tag.  

Die Vorhersage soll später für eine einfache, regelbasierte Trading-Strategie genutzt und per Backtesting evaluiert werden.

---

### Target Variable

Für jeden Handelstag \( d \) definieren wir:

\[
target(d) = 1, \text{ wenn } Close_d > Open_d
\]
\[
target(d) = 0, \text{ sonst}
\]

wobei \( Open_d \) der Eröffnungskurs und \( Close_d \) der Schlusskurs des Tages \( d \) ist.

**Interpretation:**

- `1` = Der ETF schließt höher als er eröffnet (Tagesanstieg)
- `0` = Der ETF schließt tiefer oder unverändert (kein Tagesanstieg)

→ Das Problem ist eine **binäre Klassifikation** auf Tagesebene mit **Intraday-Information** als Input.

---

### Input-Variablen (Rohdaten)

Wir verwenden **Intraday-Daten** (z. B. 1-Minuten- oder 1-Stunden-Bars) des ETFs URTH:

- Timestamp / DateTime  
- Open  
- High  
- Low  
- Close  
- Volume  

Optional werden zusätzliche Informationsquellen eingebunden:

- Historische Ereignisse / News (z. B. über *Alpaca News*)  
  - Anzahl der News pro Zeitfenster  
  - ggf. Sentiment-Scores (positiv/negativ)

Später werden aus den Rohdaten weitere Features erstellt (z. B. Intraday-Renditen, gleitende Durchschnitte, Volatilität, RSI, MACD, Volumen-Spikes, News-Features).

---

### Baseline

- **Buy & Hold** (immer investiert sein, keine aktive Intraday-Entscheidung)
