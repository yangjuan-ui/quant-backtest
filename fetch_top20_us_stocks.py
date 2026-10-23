"""US stock universe: dynamic top-20 by market cap.

Pipeline per run:
  1. Wikipedia's S&P 500 constituents table (free, static) -> candidate pool.
     Covers the vast majority of true top-20 companies; a mega-cap outside
     the S&P 500 is theoretically possible but extremely unlikely.
  2. Query each candidate's market cap via yfinance (fast_info first,
     .info as fallback), with retries, delays, a circuit breaker and a
     local cache so a rate-limited run degrades gracefully instead of
     dying.
  3. Sort locally -> top 20 -> download daily history. History download
     is layered: yfinance batch first, then per-ticker yfinance, then
     Stooq's free daily CSV as a last resort (no API key needed).

500 sequential info queries are slow; the cache (default 7-day TTL) makes
repeat runs cheap.
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TOP_N = 20
HISTORY_PERIOD = "2y"
CAP_CACHE_TTL_DAYS = 7
QUERY_DELAY = 0.25
MAX_CONSECUTIVE_CAP_FAILURES = 30  # circuit breaker for Yahoo blocks
STOOQ_URL = "https://stooq.com/q/d/l/"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0 Safari/537.36")


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
    consec_failures = 0
    for i, sym in enumerate(missing, 1):
        cap = _query_market_cap(sym)
        if cap:
            caps[sym] = cap
            consec_failures = 0
        else:
            consec_failures += 1
            log.warning("no market cap for %s (skipped)", sym)
            if consec_failures >= MAX_CONSECUTIVE_CAP_FAILURES:
                log.error("market-cap circuit breaker tripped after %d "
                          "consecutive failures (Yahoo likely blocking); "
                          "ranking on the %d caps collected so far",
                          consec_failures, len(caps))
                break
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


def _download_stooq(symbol: str) -> pd.DataFrame | None:
    """Last-resort daily history from Stooq's free CSV (no API key).

    Returns a normalized DataFrame or None. Stooq serves
    'Date,Open,High,Low,Close,Volume' for '<symbol>.us'.
    """
    try:
        r = requests.get(STOOQ_URL,
                         params={"s": f"{symbol.lower()}.us", "i": "d"},
                         headers={"User-Agent": BROWSER_UA}, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        if len(lines) < 2 or not lines[0].lower().startswith("date,open"):
            return None
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
        df = df.rename(columns={"Date": "date", "Open": "open",
                                "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.dropna(subset=["close"])
        log.info("stooq history for %s: %d bars", symbol, len(df))
        return df
    except Exception as e:  # noqa: BLE001 - fallback must never raise
        log.debug("stooq failed for %s: %s", symbol, e)
        return None


def _download_single_yf(symbol: str, period: str) -> pd.DataFrame | None:
    """Per-ticker yfinance history (works when the batch endpoint is
    blocked but single-ticker queries still answer)."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1d",
                                       auto_adjust=True, progress=False)
        df = _normalize(df)
        return df if len(df) >= 60 else None
    except Exception as e:  # noqa: BLE001 - fallback must never raise
        log.debug("yfinance single-ticker failed for %s: %s", symbol, e)
        return None


def download_history(symbols: list[str], period: str = HISTORY_PERIOD
                     ) -> dict[str, pd.DataFrame]:
    """Daily OHLCV, layered sources. Returns {symbol: DataFrame}.

    1. yfinance batch download (fast path).
    2. Any symbol missing from the batch: per-ticker yfinance.
    3. Still missing: Stooq free daily CSV.
    A symbol failing everywhere is logged and skipped, never fatal.
    """
    if not symbols:
        return {}
    out: dict[str, pd.DataFrame] = {}
    try:
        data = yf.download(symbols, period=period, interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                try:
                    df = _normalize(data.xs(sym, axis=1, level=1))
                except KeyError:
                    continue
                if len(df) >= 60:
                    out[sym] = df
        else:  # single ticker -> flat columns
            df = _normalize(data)
            if len(df) >= 60:
                out[symbols[0]] = df
        log.info("yfinance batch: %d/%d symbols", len(out), len(symbols))
    except Exception as e:
        log.warning("yfinance batch download failed (%s); "
                    "falling back per symbol", e)

    for sym in [s for s in symbols if s not in out]:
        df = _download_single_yf(sym, period)
        source = "yfinance-single"
        if df is None:
            df = _download_stooq(sym)
            source = "stooq"
        if df is not None and len(df) >= 60:
            out[sym] = df
            log.info("history for %s via %s: %d bars", sym, source, len(df))
        else:
            log.warning("no history for %s from any source", sym)
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
