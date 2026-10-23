"""Backtest engine: dynamic exits + honest aggregate metrics.

Exits are dynamic (no fixed T+7):
  trend strategy         -> ATR stop-loss / ATR take-profit, plus an exit on
                            trend reversal (fast MA crossing back below slow MA).
  mean-reversion strategy-> exit at the mean (middle Bollinger band) or when
                            RSI recovers, plus an ATR stop-loss.

Assumptions (documented, not hidden):
  * long-only, one position per symbol at a time, daily bars;
  * entry at the close of the signal bar (decision made on closed-bar data);
  * stop-loss fills at the stop price, take-profit fills at the target price;
  * a small fee_rate per side approximates commission/slippage;
  * an open position at the end of the data is force-closed (end_of_data).

Metrics reported per the project's core philosophy: win rate is NEVER shown
alone - always together with avg profit/loss ratio, expectancy and max
drawdown. A high-win-rate / low-payoff system can lose money long term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import position_size_units

EXIT_PARAMS = {
    "trend": {"stop_mult": 2.0, "target_mult": 3.0},
    "mean_reversion": {"stop_mult": 1.5, "target_mult": None},
}


def simulate(df: pd.DataFrame, entries: pd.Series, strategy: str,
             regime: pd.Series | None = None,
             initial_equity: float = 100_000.0, risk_pct: float = 0.01,
             fee_rate: float = 0.0005) -> tuple[list[dict], pd.DataFrame]:
    """Walk bars forward, applying dynamic exits.

    Returns (trades, equity_curve). equity_curve has columns [date, equity]
    with open positions marked to market.
    """
    if strategy not in EXIT_PARAMS:
        raise ValueError(f"unknown strategy: {strategy}")
    params = EXIT_PARAMS[strategy]

    trades: list[dict] = []
    curve_dates, curve_equity = [], []
    equity = initial_equity
    pos = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]

        if pos is None:
            if bool(entries.iloc[i]):
                atr = row["atr14"]
                if pd.isna(atr) or atr <= 0 or pd.isna(row["close"]):
                    pass
                else:
                    stop_dist = params["stop_mult"] * atr
                    units = position_size_units(equity, row["close"],
                                                stop_dist, risk_pct)
                    if units > 0:
                        notional = units * row["close"]
                        pos = {
                            "entry_idx": i,
                            "entry_date": date,
                            "entry_price": float(row["close"]),
                            "atr": float(atr),
                            "stop": float(row["close"] - stop_dist),
                            "target": (float(row["close"] + params["target_mult"] * atr)
                                       if params["target_mult"] else None),
                            "units": units,
                            "notional": notional,
                            "entry_fee": fee_rate * notional,
                            "regime": (str(regime.iloc[i])
                                       if regime is not None else None),
                        }
        else:
            exited, exit_price, reason = False, None, None
            if row["low"] <= pos["stop"]:
                exit_price, reason, exited = pos["stop"], "stop_loss", True
            elif (strategy == "trend" and pos["target"] is not None
                    and row["high"] >= pos["target"]):
                exit_price, reason, exited = pos["target"], "take_profit", True
            elif (strategy == "trend" and not pd.isna(row["sma_fast"])
                    and not pd.isna(row["sma_slow"])
                    and row["sma_fast"] < row["sma_slow"]):
                exit_price, reason, exited = float(row["close"]), "trend_reversal", True
            elif strategy == "mean_reversion":
                if not pd.isna(row["bb_mid"]) and row["close"] >= row["bb_mid"]:
                    exit_price, reason, exited = float(row["close"]), "mean_target", True
                elif not pd.isna(row["rsi14"]) and row["rsi14"] > 55:
                    exit_price, reason, exited = float(row["close"]), "rsi_exit", True

            if exited:
                gross = (exit_price - pos["entry_price"]) * pos["units"]
                pnl = gross - pos["entry_fee"] - fee_rate * exit_price * pos["units"]
                equity += pnl
                entry_d = pos["entry_date"]
                holding_days = (pd.Timestamp(date) - pd.Timestamp(entry_d)).days
                trades.append({
                    "strategy": strategy,
                    "regime": pos["regime"],
                    "entry_date": pd.Timestamp(entry_d).strftime("%Y-%m-%d"),
                    "entry_price": pos["entry_price"],
                    "exit_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "exit_price": float(exit_price),
                    "exit_reason": reason,
                    "holding_days": int(holding_days),
                    "position_size": pos["units"],
                    "notional": pos["notional"],
                    "atr_at_entry": pos["atr"],
                    "pnl": float(pnl),
                    "pnl_pct": float(pnl / pos["notional"]) if pos["notional"] else 0.0,
                    "win": 1 if pnl > 0 else 0,
                })
                pos = None

        mtm = equity
        if pos is not None and not pd.isna(row["close"]):
            mtm = equity + (row["close"] - pos["entry_price"]) * pos["units"]
        curve_dates.append(date)
        curve_equity.append(mtm)

    # force-close any open position at the end of the data
    if pos is not None:
        row = df.iloc[-1]
        date = df.index[-1]
        exit_price = float(row["close"])
        gross = (exit_price - pos["entry_price"]) * pos["units"]
        pnl = gross - pos["entry_fee"] - fee_rate * exit_price * pos["units"]
        equity += pnl
        entry_d = pos["entry_date"]
        holding_days = (pd.Timestamp(date) - pd.Timestamp(entry_d)).days
        trades.append({
            "strategy": strategy,
            "regime": pos["regime"],
            "entry_date": pd.Timestamp(entry_d).strftime("%Y-%m-%d"),
            "entry_price": pos["entry_price"],
            "exit_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "exit_price": exit_price,
            "exit_reason": "end_of_data",
            "holding_days": int(holding_days),
            "position_size": pos["units"],
            "notional": pos["notional"],
            "atr_at_entry": pos["atr"],
            "pnl": float(pnl),
            "pnl_pct": float(pnl / pos["notional"]) if pos["notional"] else 0.0,
            "win": 1 if pnl > 0 else 0,
        })
        curve_equity[-1] = equity

    curve = pd.DataFrame({"date": pd.to_datetime(curve_dates),
                          "equity": curve_equity})
    return trades, curve


def max_drawdown_pct(equity: pd.Series) -> float:
    """Max peak-to-trough drawdown as a positive percent."""
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0.0, np.nan)
    return float(dd.fillna(0.0).max() * 100.0)


def compute_metrics(trades: list[dict], equity_curve: pd.DataFrame,
                    initial_equity: float) -> dict:
    """Win rate + avg profit/loss ratio + expectancy + max drawdown.

    Never returns win rate without the other three.
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": None, "avg_win": None, "avg_loss": None,
            "profit_loss_ratio": None, "expectancy": None,
            "expectancy_pct": None, "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    pnl_pcts = np.array([t["pnl_pct"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative or 0
    pl_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else None
    final_equity = float(equity_curve["equity"].iloc[-1])
    return {
        "n_trades": n,
        "win_rate": float((pnls > 0).mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": pl_ratio,
        "expectancy": float(pnls.mean()),
        "expectancy_pct": float(pnl_pcts.mean()),
        "max_drawdown_pct": max_drawdown_pct(equity_curve["equity"]),
        "total_return_pct": (final_equity - initial_equity) / initial_equity * 100.0,
    }
