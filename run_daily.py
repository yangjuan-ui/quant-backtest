#!/usr/bin/env python3
"""Daily orchestrator: universe -> data -> regime/strategy -> backtest -> DB.

Usage:
    python run_daily.py                 # full daily run
    python run_daily.py --quick         # smoke test (few symbols, skip 13F)
    python run_daily.py --skip-13f      # skip the quarterly 13F layer

Each symbol is backtested independently with INITIAL_EQUITY of notional
capital per (symbol, strategy) run; the daily summary aggregates pooled
trades (win rate / profit-loss ratio / expectancy) and a combined
equal-capital equity curve (max drawdown / total return).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from database import (
    DEFAULT_DB_PATH,
    init_db,
    insert_13f_holdings,
    insert_summary,
    insert_trades,
)
from strategy import (
    add_indicators,
    detect_regime,
    mean_reversion_entries,
    trend_entries,
)
from backtest import compute_metrics, max_drawdown_pct, simulate
from fetch_top50_crypto import build_crypto_universe
from fetch_top20_us_stocks import build_us_universe, download_history
from fetch_ark_holdings import ark_ticker_universe
from fetch_13f_filings import fetch_13f_holdings

REPO_ROOT = Path(__file__).resolve().parent
INITIAL_EQUITY = 100_000.0

log = logging.getLogger("daily")


def setup_logging() -> Path:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"daily_{stamp}.log", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)
    return log_dir / f"daily_{stamp}.log"


def build_universe(quick: bool = False) -> list[tuple[str, pd.DataFrame, str, str]]:
    """[(symbol, ohlcv_df, asset_type, pool_source)]. One source failing
    never kills the whole run."""
    universe: list[tuple[str, pd.DataFrame, str, str]] = []

    try:
        crypto = build_crypto_universe()
        if quick:
            crypto = dict(list(crypto.items())[:5])
        for pair, df in crypto.items():
            universe.append((pair, df, "crypto", "top50_crypto"))
        log.info("crypto universe: %d symbols", len(crypto))
    except Exception as e:
        log.error("crypto universe failed: %s", e)

    top20_syms: set[str] = set()
    try:
        us = build_us_universe(top_n=5 if quick else 20)
        top20_syms = set(us)
        for sym, df in us.items():
            universe.append((sym, df, "us_stock", "top20_us"))
        log.info("us top20 universe: %d symbols", len(us))
    except Exception as e:
        log.error("us top20 universe failed: %s", e)

    try:
        ark_syms = ark_ticker_universe() - top20_syms
        if quick:
            ark_syms = set(sorted(ark_syms)[:5])
        # cap ARK downloads to keep the daily job within time limits
        ark_syms = set(sorted(ark_syms)[:150])
        ark_hist = download_history(sorted(ark_syms))
        for sym, df in ark_hist.items():
            universe.append((sym, df, "us_stock", "ark"))
        log.info("ark universe: %d symbols", len(ark_hist))
    except Exception as e:
        log.error("ark universe failed: %s", e)

    return universe


def run_backtests(universe, run_date: str):
    all_trades: list[dict] = []
    curves: list[pd.DataFrame] = []
    for symbol, raw_df, asset_type, pool_source in universe:
        try:
            df = add_indicators(raw_df)
            regime = detect_regime(df)
            jobs = [
                ("trend", trend_entries(df) & (regime == "trend")),
                ("mean_reversion",
                 mean_reversion_entries(df) & (regime == "range")),
            ]
            for strategy, entries in jobs:
                trades, curve = simulate(df, entries, strategy,
                                         regime=regime,
                                         initial_equity=INITIAL_EQUITY)
                for t in trades:
                    t.update({"run_date": run_date, "symbol": symbol,
                              "asset_type": asset_type,
                              "pool_source": pool_source})
                all_trades.extend(trades)
                curves.append(curve)
        except Exception as e:
            log.warning("backtest failed for %s: %s", symbol, e)
    return all_trades, curves


def combined_equity(curves: list[pd.DataFrame]) -> pd.Series:
    """Equal-capital portfolio curve: sum of per-run curves (idle capital
    counts as INITIAL_EQUITY before a run's data starts)."""
    if not curves:
        return pd.Series(dtype=float)
    frames = [c.set_index("date")["equity"] for c in curves]
    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.ffill().fillna(INITIAL_EQUITY)
    return wide.sum(axis=1)


def print_report(run_date: str, n_symbols: int, n_runs: int,
                 trades: list[dict], metrics: dict) -> None:
    def pct(x):
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    def money(x):
        return f"${x:,.0f}" if x is not None else "n/a"

    lines = [
        f"===== DAILY BACKTEST REPORT ({run_date} UTC) =====",
        f"symbols: {n_symbols} | backtests run: {n_runs}",
        f"trades closed: {metrics['n_trades']}",
        f"win rate: {pct(metrics['win_rate'])}   "
        f"avg win: {money(metrics['avg_win'])}   "
        f"avg loss: {money(metrics['avg_loss'])}",
        f"profit/loss ratio: "
        f"{metrics['profit_loss_ratio']:.2f}" if metrics["profit_loss_ratio"] is not None
        else "profit/loss ratio: n/a",
        f"expectancy/trade: {money(metrics['expectancy'])} "
        f"({pct(metrics['expectancy_pct'])})",
        f"max drawdown: -{metrics['max_drawdown_pct']:.1f}%   "
        f"total return: {metrics['total_return_pct']:+.1f}%",
    ]
    if trades:
        tdf = pd.DataFrame(trades)
        lines.append("--- by strategy ---")
        for strat, g in tdf.groupby("strategy"):
            w = (g["pnl"] > 0).mean()
            lines.append(f"  {strat:15s} n={len(g):4d} win={w * 100:5.1f}% "
                         f"expect/trade=${g['pnl'].mean():+,.0f}")
        lines.append("--- by pool ---")
        for pool, g in tdf.groupby("pool_source"):
            w = (g["pnl"] > 0).mean()
            lines.append(f"  {pool:15s} n={len(g):4d} win={w * 100:5.1f}% "
                         f"expect/trade=${g['pnl'].mean():+,.0f}")
    lines.append("=" * 52)
    report = "\n".join(lines)
    log.info("\n" + report)
    print(report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: few symbols, skip 13F")
    ap.add_argument("--skip-13f", action="store_true",
                    help="skip the quarterly 13F reference layer")
    args = ap.parse_args()

    setup_logging()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info("daily run starting (%s UTC)", run_date)
    init_db(DEFAULT_DB_PATH)

    # --- quarterly 13F reference layer (supplementary, not a daily signal)
    if not args.skip_13f and not args.quick:
        try:
            holdings = fetch_13f_holdings()
            n = insert_13f_holdings(holdings, run_date)
            log.info("13F reference layer stored: %d rows", n)
            for h in holdings[:10]:
                log.info("  13F top: %-40s $%13,.0f (filed %s)",
                         h["issuer"][:40], h["value_usd"], h["filing_date"])
        except Exception as e:
            log.error("13F layer failed (non-fatal): %s", e)
    else:
        log.info("13F layer skipped")

    universe = build_universe(quick=args.quick)
    if not universe:
        log.critical("no symbols in universe - aborting run")
        return 1
    log.info("total universe: %d symbols", len(universe))

    all_trades, curves = run_backtests(universe, run_date)
    n_runs = len(curves)
    log.info("backtests finished: %d runs, %d closed trades",
             n_runs, len(all_trades))

    inserted = insert_trades(all_trades)
    log.info("inserted %d trades into %s", inserted, DEFAULT_DB_PATH)

    eq = combined_equity(curves)
    metrics = compute_metrics(all_trades, pd.DataFrame(
        {"date": eq.index, "equity": eq.values}), INITIAL_EQUITY * max(n_runs, 1))
    # compute_metrics expects n_trades etc.; patch portfolio-level fields
    metrics["max_drawdown_pct"] = max_drawdown_pct(eq) if len(eq) else 0.0
    if len(eq):
        metrics["total_return_pct"] = (
            (eq.iloc[-1] - INITIAL_EQUITY * n_runs) / (INITIAL_EQUITY * n_runs) * 100.0
        )

    summary = {"run_date": run_date, "n_symbols": len(universe),
               "n_trades": metrics["n_trades"],
               "win_rate": metrics["win_rate"], "avg_win": metrics["avg_win"],
               "avg_loss": metrics["avg_loss"],
               "profit_loss_ratio": metrics["profit_loss_ratio"],
               "expectancy": metrics["expectancy"],
               "expectancy_pct": metrics["expectancy_pct"],
               "max_drawdown_pct": metrics["max_drawdown_pct"],
               "total_return_pct": metrics["total_return_pct"]}
    insert_summary(summary)
    print_report(run_date, len(universe), n_runs, all_trades, metrics)
    log.info("daily run done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
