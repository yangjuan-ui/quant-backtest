"""ARK holdings (daily): standalone, pluggable module.

ARK Invest publishes a full holdings CSV per ETF every trading day, free and
without registration. ARK changes these download URLs from time to time, so:

  * all URLs live in ARK_CSV_URLS below - update them in ONE place;
  * any download/parse failure raises a clear error telling a human to check
    ark-funds.com for the new URL pattern. The module never pretends the
    links are eternal.

    *** URL PATTERN VERIFIED 2026-10-22 ***
    Current working pattern (confirmed for ARKK on 2026-10-22):
        https://assets.ark-funds.com/fund-documents/funds-etf-csv/<FUND>_<TICKER>_HOLDINGS.csv
    CSV columns: date,fund,company,ticker,cusip,shares,market value ($),weight (%)
    ARK still changes these URLs from time to time, so the pluggable design
    and the clear human-actionable error in fetch_ark_holdings() stay in
    place regardless.
"""
from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARK_CSV_URLS = {
    "ARKK": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKW": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_NEXT_GENERATION_INTERNET_ARKW_HOLDINGS.csv",
    "ARKQ": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_AUTONOMOUS_TECHNOLOGY_AND_ROBOTICS_ARKQ_HOLDINGS.csv",
    "ARKF": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_FINTECH_INNOVATION_ARKF_HOLDINGS.csv",
    "ARKG": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_GENOMIC_REVOLUTION_ARKG_HOLDINGS.csv",
    "ARKX": "https://assets.ark-funds.com/fund-documents/funds-etf-csv/ARK_SPACE_EXPLORATION_AND_INNOVATION_ARKX_HOLDINGS.csv",
}

TICKER_CANDIDATES = {"ticker", "symbol"}


def _download_csv(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "quant-backtest/1.0"})
    r.raise_for_status()
    return r.text


def _extract_tickers(csv_text: str) -> set[str]:
    df = pd.read_csv(StringIO(csv_text))
    cols = {str(c).strip().lower(): c for c in df.columns}
    tick_col = next((cols[k] for k in TICKER_CANDIDATES if k in cols), None)
    if tick_col is None:
        raise ValueError(f"ticker column not found; columns={list(df.columns)}")
    tickers = (
        df[tick_col].astype(str).str.strip().str.upper()
        .str.replace(".", "-", regex=False)
    )
    return {t for t in tickers if t and t != "NAN"}


def fetch_ark_holdings(urls: dict[str, str] | None = None) -> dict[str, set[str]]:
    """Return {ETF: set of tickers}. Raises RuntimeError on failure with
    human-actionable guidance (per design: never silently pretend)."""
    urls = urls or ARK_CSV_URLS
    holdings: dict[str, set[str]] = {}
    failures: list[str] = []
    for etf, url in urls.items():
        try:
            tickers = _extract_tickers(_download_csv(url))
            holdings[etf] = tickers
            log.info("%s: %d holdings", etf, len(tickers))
        except Exception as e:
            failures.append(f"{etf}: {e}")
            log.error("%s download/parse failed: %s", etf, e)
    if failures:
        raise RuntimeError(
            "ARK holdings fetch failed for: " + "; ".join(failures) + "\n"
            "ACTION REQUIRED: ARK likely changed its CSV URL pattern. "
            "Open https://ark-funds.com, find the current per-ETF holdings "
            "CSV download link, and update ARK_CSV_URLS in fetch_ark_holdings.py."
        )
    return holdings


def ark_ticker_universe(holdings: dict[str, set[str]] | None = None) -> set[str]:
    """Deduplicated ticker universe across all ARK ETFs."""
    holdings = holdings if holdings is not None else fetch_ark_holdings()
    universe: set[str] = set()
    for tickers in holdings.values():
        universe |= tickers
    return universe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    uni = ark_ticker_universe()
    print(f"ARK universe: {len(uni)} tickers")
