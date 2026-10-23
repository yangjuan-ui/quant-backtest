"""Crypto universe: CoinGecko top-50 by market cap -> ccxt OHLCV.

Pipeline per run:
  1. CoinGecko free API /coins/markets (market_cap_desc) -> current top 50.
  2. Pick a working ccxt spot exchange. GitHub-hosted runners sit on US
     IPs, where Binance answers 451 (restricted location), so the code
     tries US-accessible exchanges first: kraken (USDT pairs), then
     coinbase (USD pairs), with binance kept as a last resort.
  3. Map each coin to a spot pair on the chosen exchange (e.g. BTC/USDT).
     Coins with no liquid pair are SKIPPED WITH A LOG, never fatal.
  4. Download daily OHLCV via ccxt (rate-limit aware).

CoinGecko free tier is rate-limited (~10-30 req/min): delays and retries
are built in.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import ccxt
import pandas as pd
import requests

log = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
# (exchange_id, quote_currency) tried in order until one answers.
# Binance is last: it 451-blocks US-based runner IPs.
EXCHANGE_CANDIDATES: list[tuple[str, str]] = [
    ("kraken", "USDT"),
    ("coinbase", "USD"),
    ("binance", "USDT"),
]
OHLCV_LIMIT = 400          # ~400 daily bars per symbol
CG_DELAY = 12.0            # seconds between CoinGecko calls on 429 backoff base
CCXT_DELAY = 0.4           # seconds between OHLCV calls (on top of ccxt rate limit)


def get_top50(vs_currency: str = "usd", per_page: int = 50,
              retries: int = 4) -> list[dict]:
    """Return current top-N coins by market cap from CoinGecko."""
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "false",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(COINGECKO_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = CG_DELAY * (attempt + 1)
                log.warning("CoinGecko 429 - backing off %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            log.warning("CoinGecko request failed (attempt %d): %s",
                        attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"CoinGecko top-50 fetch failed after {retries} tries: {last_err}")


def pick_exchange(candidates: list[tuple[str, str]] = EXCHANGE_CANDIDATES
                  ) -> tuple[object, str]:
    """Return (exchange_instance, quote) for the first reachable candidate.

    Each candidate is tried with load_markets(); geo-blocked or otherwise
    unreachable exchanges are skipped with a log line, never fatal here.
    Raises RuntimeError only if every candidate fails.
    """
    last_err = None
    for exchange_id, quote in candidates:
        try:
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            exchange.load_markets()
            log.info("using ccxt exchange %s (quote %s)", exchange_id, quote)
            return exchange, quote
        except Exception as e:  # noqa: BLE001 - any failure -> try next
            last_err = e
            log.warning("ccxt exchange %s unreachable (%s); trying next",
                        exchange_id, e)
    raise RuntimeError(f"no ccxt exchange reachable: {last_err}")


def map_to_ccxt_symbols(coins: list[dict], exchange,
                        quote: str) -> tuple[list[tuple[dict, str]], list[dict]]:
    """Map CoinGecko coins to '<BASE>/<QUOTE>' spot pairs on an exchange.

    Returns (mapped, skipped). Skipped coins are logged, not fatal.
    """
    markets = exchange.load_markets()
    exchange_id = getattr(exchange, "id", "?")
    mapped, skipped = [], []
    for coin in coins:
        base = str(coin.get("symbol", "")).upper()
        pair = f"{base}/{quote}"
        if pair in markets and markets[pair].get("spot", True):
            mapped.append((coin, pair))
        else:
            skipped.append(coin)
            log.info("skip %s (%s): no %s spot pair on %s",
                     coin.get("name"), base, quote, exchange_id)
    return mapped, skipped


def fetch_ohlcv(exchange, pair: str, timeframe: str = "1d",
                limit: int = OHLCV_LIMIT) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def build_crypto_universe(save_dir: str | Path | None = None,
                          exchange_id: str | None = None
                          ) -> dict[str, pd.DataFrame]:
    """Full pipeline -> {pair: OHLCV DataFrame}. Saves CSVs if save_dir given.

    exchange_id pins a single exchange (skipping the fallback chain), e.g.
    for local testing; when None the EXCHANGE_CANDIDATES chain is used.
    """
    coins = get_top50()
    log.info("CoinGecko returned %d coins", len(coins))

    candidates = ([(exchange_id, "USDT")] if exchange_id
                  else EXCHANGE_CANDIDATES)
    exchange, quote = pick_exchange(candidates)
    mapped, skipped = map_to_ccxt_symbols(coins, exchange, quote)
    log.info("mapped %d pairs, skipped %d", len(mapped), len(skipped))

    universe: dict[str, pd.DataFrame] = {}
    for _, pair in mapped:
        try:
            df = fetch_ohlcv(exchange, pair)
            if len(df) < 60:
                log.info("skip %s: only %d bars", pair, len(df))
                continue
            universe[pair] = df
            log.info("fetched %s: %d bars", pair, len(df))
        except Exception as e:  # per-symbol failure must not kill the run
            log.warning("failed %s: %s", pair, e)
        time.sleep(CCXT_DELAY)

    if save_dir and universe:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for pair, df in universe.items():
            fname = pair.replace("/", "_") + ".csv"
            df.to_csv(save_dir / fname)
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    uni = build_crypto_universe(save_dir=Path(__file__).parent / "data" / "crypto")
    print(f"universe size: {len(uni)}")
