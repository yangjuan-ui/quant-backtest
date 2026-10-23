"""Strategy layer: explainable, layered, non-black-box.

Layer 1 - Regime filter:  trend market vs. range market.
Layer 2A - Trend market:   dual-MA cross + ADX confirmation + ATR stop/take-profit.
                           (Typical win rate 30-45%; profits come from the
                           profit/loss ratio: cut losses, let winners run.)
Layer 2B - Range market:   RSI overbought/oversold + Bollinger touch + exit at
                           the mean (middle band). (Typical win rate 55-70%;
                           smaller payoff per trade.)
Layer 3 - Position sizing: ATR-based (higher volatility -> smaller size),
                           never a fixed size.

Indicators are pure pandas/numpy so the repo has no TA-Lib dependency.
Long-only, daily bars. Microstructure signals (order-flow imbalance etc.)
are deliberately NOT implemented: free data sources lack tick granularity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- indicators

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. 100 when there were no down moves, 0 when no up moves."""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    out = 100.0 - 100.0 / (1.0 + ru / rd)
    out = out.where(rd != 0, 100.0)
    out = out.where(ru != 0, 0.0)
    return out


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX."""
    h, l = df["high"], df["low"]
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index
    )
    pc = df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / n, min_periods=n, adjust=False).mean() / atr_s
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * std, mid - k * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all indicator columns added."""
    df = df.copy()
    df["sma_fast"] = sma(df["close"], 20)
    df["sma_slow"] = sma(df["close"], 50)
    df["atr14"] = atr(df, 14)
    df["rsi14"] = rsi(df["close"], 14)
    df["adx14"] = adx(df, 14)
    mid, up, lo = bollinger(df["close"], 20, 2.0)
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, up, lo
    # ATR ratio vs its own history: alternative regime input
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(60, min_periods=60).mean()
    return df


# ------------------------------------------------------------ regime filter

def detect_regime(df: pd.DataFrame, method: str = "adx",
                  adx_trend: float = 20.0) -> pd.Series:
    """Classify each bar as 'trend' or 'range'.

    method='adx'       (default): trend when ADX(14) > adx_trend.
    method='atr_ratio' (alternative): trend when ATR(14) is expanding
                       relative to its 60-bar average (ratio > 1).
    Warmup bars default to 'range' (the conservative choice).
    """
    if method == "atr_ratio":
        regime = pd.Series(
            np.where(df["atr_ratio"] > 1.0, "trend", "range"), index=df.index
        )
    else:
        regime = pd.Series(
            np.where(df["adx14"] > adx_trend, "trend", "range"), index=df.index
        )
    # unknown warmup -> range
    mask = df["adx14"].isna() if method == "adx" else df["atr_ratio"].isna()
    regime[mask] = "range"
    return regime


# ------------------------------------------------------------------ entries

def trend_entries(df: pd.DataFrame, adx_confirm: float = 25.0,
                  cross_lookback: int = 5) -> pd.Series:
    """Long entry: fast MA crossed above slow MA *recently* and ADX confirms.

    The cross and the ADX confirmation rarely land on the same bar (ADX
    lags), so a cross within the last `cross_lookback` bars counts, as long
    as price is still in the uptrend (fast MA above slow MA).
    """
    cross = (df["sma_fast"] > df["sma_slow"]) & (
        df["sma_fast"].shift(1) <= df["sma_slow"].shift(1)
    )
    recent_cross = cross.rolling(cross_lookback, min_periods=1).max() > 0
    in_uptrend = df["sma_fast"] > df["sma_slow"]
    return (recent_cross & in_uptrend & (df["adx14"] > adx_confirm)).fillna(False)


def mean_reversion_entries(df: pd.DataFrame,
                           rsi_oversold: float = 30.0) -> pd.Series:
    """Long entry: RSI oversold and price touches the lower Bollinger band."""
    return ((df["rsi14"] < rsi_oversold) & (df["close"] <= df["bb_lo"])).fillna(False)


# ---------------------------------------------------------- position sizing

def position_size_units(equity: float, entry_price: float, stop_distance: float,
                        risk_pct: float = 0.01,
                        max_position_pct: float = 0.20) -> float:
    """ATR-based sizing: risk a fixed fraction of equity per trade.

    units = (equity * risk_pct) / stop_distance, capped so one position
    never exceeds max_position_pct of equity (protects against tiny-ATR
    assets producing absurd size).
    """
    if equity <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0.0
    risk_units = equity * risk_pct / stop_distance
    cap_units = equity * max_position_pct / entry_price
    return max(0.0, min(risk_units, cap_units))
