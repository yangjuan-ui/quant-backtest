"""SQLite persistence layer for the quant backtest system.

Tables:
  trades       - one row per closed simulated trade, with signal-source and
                 pool-source tags (never judge a strategy by win rate alone).
  run_summary  - per-day aggregate: win rate + avg profit/loss ratio +
                 expectancy + max drawdown.
  holdings_13f - quarterly 13F reference layer (SEC EDGAR). Stored separately
                 on purpose: 13F is a lagged quarterly reference, not a
                 daily trading signal.

The database file lives at db/trading_records.db inside the repo and is
committed back by the GitHub Actions workflow after every run, so git
history itself is the permanent, traceable record.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = REPO_ROOT / "db" / "trading_records.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT NOT NULL,   -- YYYY-MM-DD (UTC) of the daily run
    symbol         TEXT NOT NULL,   -- e.g. BTC/USDT or AAPL
    asset_type     TEXT NOT NULL,   -- crypto | us_stock
    pool_source    TEXT NOT NULL,   -- top50_crypto | top20_us | ark | 13f
    strategy       TEXT NOT NULL,   -- trend | mean_reversion
    regime         TEXT,            -- trend | range (regime at entry)
    entry_date     TEXT,
    entry_price    REAL,
    exit_date      TEXT,
    exit_price     REAL,
    exit_reason    TEXT,            -- take_profit | stop_loss | mean_target |
                                   -- rsi_exit | trend_reversal | end_of_data
    holding_days   INTEGER,
    position_size  REAL,            -- base units
    notional       REAL,            -- entry_price * position_size
    atr_at_entry   REAL,
    pnl            REAL,            -- net of fees
    pnl_pct        REAL,            -- pnl / notional
    win            INTEGER,         -- 1 = win, 0 = loss
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_summary (
    run_date           TEXT PRIMARY KEY,
    n_symbols          INTEGER,
    n_trades           INTEGER,
    win_rate           REAL,
    avg_win            REAL,
    avg_loss           REAL,
    profit_loss_ratio  REAL,        -- avg_win / |avg_loss|
    expectancy         REAL,        -- mean net pnl per trade
    expectancy_pct     REAL,        -- mean pnl_pct per trade
    max_drawdown_pct   REAL,
    total_return_pct   REAL,
    created_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdings_13f (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT NOT NULL,
    fund           TEXT NOT NULL,
    cik            TEXT NOT NULL,
    filing_date    TEXT,
    report_period  TEXT,            -- quarter end date from the filing
    issuer         TEXT,
    cusip          TEXT,
    value_usd      REAL,            -- market value in USD
    shares         REAL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_date);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_13f_run ON holdings_13f(run_date);
"""

TRADE_COLUMNS = [
    "run_date", "symbol", "asset_type", "pool_source", "strategy", "regime",
    "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason",
    "holding_days", "position_size", "notional", "atr_at_entry",
    "pnl", "pnl_pct", "win",
]


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create tables/indexes if they do not exist. Safe to call every run."""
    db_path = Path(db_path)
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
    return db_path


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def insert_trades(trades: list[dict], db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Insert closed-trade dicts. Returns number of rows inserted."""
    if not trades:
        return 0
    now = _now_utc()
    rows = [
        tuple(t.get(c) for c in TRADE_COLUMNS) + (now,)
        for t in trades
    ]
    cols = ", ".join(TRADE_COLUMNS + ["created_at"])
    placeholders = ", ".join(["?"] * (len(TRADE_COLUMNS) + 1))
    with _connect(db_path) as conn:
        conn.executemany(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})", rows
        )
    return len(rows)


def insert_summary(summary: dict, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    now = _now_utc()
    cols = [
        "run_date", "n_symbols", "n_trades", "win_rate", "avg_win", "avg_loss",
        "profit_loss_ratio", "expectancy", "expectancy_pct",
        "max_drawdown_pct", "total_return_pct", "created_at",
    ]
    with _connect(db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO run_summary ({', '.join(cols)}) "
            f"VALUES ({', '.join(['?'] * len(cols))})",
            tuple(summary.get(c) for c in cols[:-1]) + (now,),
        )


def insert_13f_holdings(
    holdings: list[dict], run_date: str, db_path: str | Path = DEFAULT_DB_PATH
) -> int:
    if not holdings:
        return 0
    now = _now_utc()
    cols = ["run_date", "fund", "cik", "filing_date", "report_period",
            "issuer", "cusip", "value_usd", "shares", "created_at"]
    rows = [
        (run_date, h.get("fund"), h.get("cik"), h.get("filing_date"),
         h.get("report_period"), h.get("issuer"), h.get("cusip"),
         h.get("value_usd"), h.get("shares"), now)
        for h in holdings
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            f"INSERT INTO holdings_13f ({', '.join(cols)}) "
            f"VALUES ({', '.join(['?'] * len(cols))})",
            rows,
        )
    return len(rows)


def load_trades_df(db_path: str | Path = DEFAULT_DB_PATH,
                   run_date: str | None = None):
    """Return trades as a pandas DataFrame (pandas imported lazily)."""
    import pandas as pd

    query = "SELECT * FROM trades"
    params: tuple = ()
    if run_date:
        query += " WHERE run_date = ?"
        params = (run_date,)
    with _connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)
