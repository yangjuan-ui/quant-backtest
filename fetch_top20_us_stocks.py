"""US stock universe: dynamic top-20 by market cap.

Pipeline per run:
  1. Wikipedia's S&P 500 constituents table (free, static) -> candidate pool.
     Covers the vast majority of true top-20 companies; a mega-cap outside
     the S&P 500 is theoretically possible but extremely unlikely.
  2. Query each candidate's market cap via yfinance (fast_info first,
     .info as fallback), with retries, delays and a local cache so a
     rate-limited run degrades gracefully instead of dying.
  3. Sort locally -> top 20 -> download daily history.

500 sequential info queries are slow; the cache (default 7-day TTL) makes
repeat runs cheap.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TOP_N = 20
HISTORY_PERIOD = "2y"
CAP_CACHE_TTL_DAYS = 7
QUERY_DELAY = 0.25


def get_sp500_symbols() -> list[str]:
    """Candidate pool from Wikipedia's S&P 500 table."""
    tables = pd.read_html(WIKI_URL)
    df = tables[0]
    sym_col = next(c for c in df.columns if "symbol" in str(c).lower())
    syms = (
        df[sym_col].astype(str).str.strip()
        .str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (Yahoo format)
        .str.split().str[0]                  # drop footnote artifacts
    )
    symbols = sorted({s for s in syms if s and s != "nan"})
    log.info("S&P 500 candidate pool: %d symbols", len(symbols))
    return symbols


def _query_market_cap(symbol: str, retries: int = 3) -> float | None:
    t = yf.Ticker(symbol)
    for attempt in range(retries):
        try:
            cap = t.fast_info.get("marketCap") if hasattr(t, "fast_info") else None
            if cap:
                return float(cap)
        except Exception:
            pass
        try:
            cap = t.info.get("marketCap")
            if cap:
                return float(cap)
        except Exception as e:
            log.debug("%s attempt %d: %s", symbol, attempt + 1, e)
        time.sleep(1.5 * (attempt + 1))
    return None


def get_market_caps(symbols: list[str],
                    cache_path: str | Path = "data/market_caps.json",
                    ttl_days: int = CAP_CACHE_TTL_DAYS) -> dict[str, float]:
    """Market caps with a JSON cache; stale/missing entries are refreshed."""
    cache_path = Path(cache_path)
    cache: dict = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            age_days = (datetime.now(timezone.utc)
                        - datetime.fromisoformat(payload["asof"])).days
            if age_days < ttl_days:
                cache = payload["caps"]
                log.info("using market-cap cache (%d entries, %d days old)",
                         len(cache), age_days)
        except Exception as e:
            log.warning("ignoring corrupt cap cache: %s", e)

    caps = dict(cache)
    missing = [s for s in symbols if s not in caps]
    for i, sym in enumerate(missing, 1):
        cap = _query_market_cap(sym)
        if cap:
            caps[sym] = cap
        else:
            log.warning("no market cap for %s (skipped)", sym)
        if i % 50 == 0:
            log.info("market-cap progress %d/%d", i, len(missing))
        time.sleep(QUERY_DELAY)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"asof": datetime.now(timezone.utc).isoformat(), "caps": caps}))
    return caps


def get_top_n_by_market_cap(caps: dict[str, float], n: int = TOP_N) -> list[str]:
    ranked = sorted(caps.items(), key=lambda kv: kv[1], reverse=True)
    top = [s for s, _ in ranked[:n]]
    log.info("top %d by market cap: %s", n, ", ".join(top))
    return top


def download_history(symbols: list[str], period: str = HISTORY_PERIOD
                     ) -> dict[str, pd.DataFrame]:
    """Daily OHLCV via yfinance. Returns {symbol: DataFrame}."""
    if not symbols:
        return {}
    data = yf.download(symbols, period=period, interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                df = data.xs(sym, axis=1, level=1)
            except KeyError:
                log.warning("no history for %s", sym)
                continue
            df = _normalize(df)
            if len(df) >= 60:
                out[sym] = df
    else:  # single ticker -> flat columns
        df = _normalize(data)
        if len(df) >= 60:
            out[symbols[0]] = df
    log.info("downloaded history for %d/%d symbols", len(out), len(symbols))
    return out


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_us_universe(top_n: int = TOP_N,
                      save_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    symbols = get_sp500_symbols()
    caps = get_market_caps(symbols)
    top = get_top_n_by_market_cap(caps, top_n)
    universe = download_history(top)
    if save_dir and universe:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for sym, df in universe.items():
            df.to_csv(save_dir / f"{sym}.csv")
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    uni = build_us_universe(save_dir=Path(__file__).parent / "data" / "us_stocks")
    print(f"universe size: {len(uni)}")
