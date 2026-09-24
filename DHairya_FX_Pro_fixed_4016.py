
"""
DHairya FX Pro
Single-file Streamlit market-analysis and paper-trading terminal.

AI:
- Uses Groq's OpenAI-compatible API endpoint.
- API key is read from Streamlit secrets/environment; it is intentionally NOT
  embedded in source code.
- Current default model is configurable with GROQ_MODEL. Groq's current public
  model catalogue should be checked before selecting a specific model.

Market data:
- Yahoo Finance via yfinance. Data can be delayed/limited and is not a
  professional exchange feed.
- All orders are simulated paper trades. No broker execution is implemented.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DHairya FX Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "DHairya FX Pro"
DB_PATH = Path(__file__).resolve().with_name("dhairyafxpro.db")

# Groq model is configurable. Do not hard-code an API key.
# Groq's model catalogue changes over time; this is a currently supported
# general-purpose model and can be changed in secrets/env without editing code.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYMBOLS: Dict[str, Dict[str, str]] = {
    "EURUSD": {"ticker": "EURUSD=X", "name": "EUR/USD", "category": "Forex"},
    "GBPUSD": {"ticker": "GBPUSD=X", "name": "GBP/USD", "category": "Forex"},
    "USDJPY": {"ticker": "JPY=X", "name": "USD/JPY", "category": "Forex"},
    "AUDUSD": {"ticker": "AUDUSD=X", "name": "AUD/USD", "category": "Forex"},
    "USDCAD": {"ticker": "CAD=X", "name": "USD/CAD", "category": "Forex"},
    "USDCHF": {"ticker": "CHF=X", "name": "USD/CHF", "category": "Forex"},
    "NZDUSD": {"ticker": "NZDUSD=X", "name": "NZD/USD", "category": "Forex"},
    "GOLD": {"ticker": "GC=F", "name": "Gold / XAUUSD", "category": "Metals"},
    "SILVER": {"ticker": "SI=F", "name": "Silver", "category": "Metals"},
    "COPPER": {"ticker": "HG=F", "name": "Copper", "category": "Metals"},
    "CRUDE": {"ticker": "CL=F", "name": "Crude Oil / WTI", "category": "Energy"},
    "BRENT": {"ticker": "BZ=F", "name": "Brent Crude", "category": "Energy"},
    "NATGAS": {"ticker": "NG=F", "name": "Natural Gas", "category": "Energy"},
    "BTC": {"ticker": "BTC-USD", "name": "Bitcoin", "category": "Crypto"},
    "ETH": {"ticker": "ETH-USD", "name": "Ethereum", "category": "Crypto"},
    "SOL": {"ticker": "SOL-USD", "name": "Solana", "category": "Crypto"},
    "SP500": {"ticker": "^GSPC", "name": "S&P 500", "category": "Indices"},
    "NASDAQ": {"ticker": "^IXIC", "name": "NASDAQ Composite", "category": "Indices"},
    "DOW": {"ticker": "^DJI", "name": "Dow Jones", "category": "Indices"},
    "NIFTY": {"ticker": "^NSEI", "name": "NIFTY 50", "category": "Indices"},
    "BANKNIFTY": {"ticker": "^NSEBANK", "name": "BANK NIFTY", "category": "Indices"},
    "DAX": {"ticker": "^GDAXI", "name": "DAX", "category": "Indices"},
}

ALIASES = {
    "EUR/USD": "EURUSD",
    "EURUSD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "GBPUSD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "USDJPY": "USDJPY",
    "AUD/USD": "AUDUSD",
    "AUDUSD": "AUDUSD",
    "USD/CAD": "USDCAD",
    "USDCAD": "USDCAD",
    "USD/CHF": "USDCHF",
    "USDCHF": "USDCHF",
    "NZD/USD": "NZDUSD",
    "NZDUSD": "NZDUSD",
    "XAUUSD": "GOLD",
    "XAU": "GOLD",
    "GOLD": "GOLD",
    "SILVER": "SILVER",
    "XAGUSD": "SILVER",
    "COPPER": "COPPER",
    "WTI": "CRUDE",
    "CRUDE": "CRUDE",
    "OIL": "CRUDE",
    "BRENT": "BRENT",
    "NATGAS": "NATGAS",
    "NG": "NATGAS",
    "BTC": "BTC",
    "BITCOIN": "BTC",
    "BTCUSD": "BTC",
    "ETH": "ETH",
    "ETHEREUM": "ETH",
    "SOL": "SOL",
    "SOLANA": "SOL",
    "SP500": "SP500",
    "S&P500": "SP500",
    "S&P 500": "SP500",
    "SPX": "SP500",
    "NASDAQ": "NASDAQ",
    "NAS100": "NASDAQ",
    "NDX": "NASDAQ",
    "DOW": "DOW",
    "DJI": "DOW",
    "DOWJONES": "DOW",
    "NIFTY": "NIFTY",
    "NIFTY50": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "DAX": "DAX",
}

TIMEFRAMES = {
    "1m": {"interval": "1m", "period": "7d", "minutes": 1},
    "5m": {"interval": "5m", "period": "30d", "minutes": 5},
    "15m": {"interval": "15m", "period": "60d", "minutes": 15},
    "1H": {"interval": "60m", "period": "730d", "minutes": 60},
    "4H": {"interval": "60m", "period": "730d", "minutes": 240},
    "1D": {"interval": "1d", "period": "5y", "minutes": 1440},
    "1W": {"interval": "1wk", "period": "10y", "minutes": 10080},
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
:root {
    --bg: #080b12;
    --panel: #101520;
    --panel2: #151b27;
    --border: #242c3a;
    --text: #edf2f7;
    --muted: #8d99aa;
    --green: #28d17c;
    --red: #ff5d73;
    --blue: #4da3ff;
    --gold: #f6c85f;
}
.stApp {
    background: radial-gradient(circle at 15% 0%, #131d31 0%, var(--bg) 34%);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: #0b1019;
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1800px; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20,27,40,.95), rgba(13,18,28,.95));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
}
[data-testid="stMetricValue"] { font-weight: 700; }
div[data-testid="stVerticalBlock"] > div:has(> div.stButton) button {
    border-radius: 10px;
}
button[kind="primary"] {
    background: linear-gradient(90deg, #1668d8, #248cff);
    border: 0;
}
.fx-card {
    background: linear-gradient(145deg, rgba(17,23,35,.98), rgba(11,15,23,.98));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,.16);
}
.fx-title { font-size: 30px; font-weight: 800; letter-spacing: -.5px; }
.fx-subtitle { color: var(--muted); margin-top: -8px; }
.badge {
    display:inline-block; padding:4px 9px; border-radius:999px;
    font-size:11px; font-weight:700; letter-spacing:.4px;
    background:#1b2433; border:1px solid #2b3748;
}
.badge-green { color:#35e58a; }
.badge-red { color:#ff6b7f; }
.badge-blue { color:#69b5ff; }
.small-muted { color: var(--muted); font-size: 12px; }
</style>
"""

LIGHT_CSS = """
<style>
:root { --bg:#f5f7fb; --panel:#ffffff; --border:#dfe4ec; --text:#172033; --muted:#697386; }
.stApp { background:#f5f7fb; color:#172033; }
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #dfe4ec; }
.block-container { padding-top:1.1rem; max-width:1800px; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #dfe4ec; border-radius:14px; padding:12px 14px; }
.fx-card { background:#ffffff; border:1px solid #dfe4ec; border-radius:16px; padding:16px; margin-bottom:12px; }
.fx-subtitle,.small-muted { color:#697386; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; font-size:11px; font-weight:700; background:#eef2f7; border:1px solid #dfe4ec; }
.badge-green { color:#168a4e; } .badge-red { color:#c83449; } .badge-blue { color:#1c6cb8; }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "selected_symbol": "EURUSD",
        "selected_tf": "1H",
        "theme": "Dark",
        "chart_type": "Candlestick",
        "auto_refresh": False,
        "risk_pct": 1.0,
        "account_size": 10000.0,
        "ai_result": "",
        "chat_history": [],
        "last_command": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
st.markdown(DARK_CSS if st.session_state.theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db_connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            pnl REAL NOT NULL DEFAULT 0,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            close_price REAL,
            close_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


def get_trades() -> pd.DataFrame:
    conn = db_connect()
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
    finally:
        conn.close()
    return df


def open_paper_trade(
    symbol: str,
    side: str,
    quantity: float,
    entry: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> None:
    conn = db_connect()
    conn.execute(
        """
        INSERT INTO trades
        (symbol, side, quantity, entry, stop_loss, take_profit, status, pnl, opened_at)
        VALUES (?, ?, ?, ?, ?, ?, 'OPEN', 0, ?)
        """,
        (
            symbol,
            side,
            quantity,
            entry,
            stop_loss,
            take_profit,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def close_trade(trade_id: int, close_price: float, reason: str) -> bool:
    conn = db_connect()
    row = conn.execute("SELECT * FROM trades WHERE id=? AND status='OPEN'", (trade_id,)).fetchone()
    if row is None:
        conn.close()
        return False

    if row["side"] == "BUY":
        pnl = (close_price - row["entry"]) * row["quantity"]
    else:
        pnl = (row["entry"] - close_price) * row["quantity"]

    conn.execute(
        """
        UPDATE trades
        SET status='CLOSED', pnl=?, closed_at=?, close_price=?, close_reason=?
        WHERE id=?
        """,
        (pnl, datetime.now(timezone.utc).isoformat(), close_price, reason, trade_id),
    )
    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]

    rename_map = {str(c).strip().title(): c for c in out.columns}
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in out.columns:
            # Some yfinance outputs may have lowercase columns.
            candidate = next((c for c in out.columns if str(c).lower() == col.lower()), None)
            if candidate is not None:
                out[col] = out[candidate]

    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame()

    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)

    return out


@st.cache_data(ttl=45, show_spinner=False)
def fetch_yahoo_raw(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    result = df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return result


@st.cache_data(ttl=45, show_spinner=False)
def get_market_data(symbol: str, timeframe: str) -> pd.DataFrame:
    meta = SYMBOLS[symbol]
    tf = TIMEFRAMES[timeframe]

    raw = fetch_yahoo_raw(meta["ticker"], tf["period"], tf["interval"])
    if raw.empty:
        return raw

    if timeframe == "4H":
        # Yahoo generally provides 60m bars; aggregate locally.
        return resample_ohlcv(raw, "4h")

    return raw


@st.cache_data(ttl=45, show_spinner=False)
def get_watch_data(symbol: str) -> pd.DataFrame:
    meta = SYMBOLS[symbol]
    return fetch_yahoo_raw(meta["ticker"], "5d", "1d")


def current_price(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df["Close"].iloc[-1])


def pct_change(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return float("nan")
    previous = float(df["Close"].iloc[-2])
    latest = float(df["Close"].iloc[-1])
    return ((latest - previous) / previous * 100.0) if previous else float("nan")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal = line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = line - signal
    return line, signal, hist


def bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    volume = df["Volume"].replace(0, np.nan)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    result = (typical * volume).cumsum() / volume.cumsum()
    return result.ffill()


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg = df["Volume"].rolling(period, min_periods=period).mean()
    return (df["Volume"] / avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    opens = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        opens.append((opens[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2.0)
    ha["HA_Open"] = opens
    ha["HA_High"] = pd.concat([df["High"], ha["HA_Open"], ha["HA_Close"]], axis=1).max(axis=1)
    ha["HA_Low"] = pd.concat([df["Low"], ha["HA_Open"], ha["HA_Close"]], axis=1).min(axis=1)
    return ha


# ---------------------------------------------------------------------------
# Structure / SMC approximations
# ---------------------------------------------------------------------------

def swing_points(df: pd.DataFrame, window: int = 3) -> Tuple[pd.Series, pd.Series]:
    highs = df["High"].eq(df["High"].rolling(2 * window + 1, center=True).max())
    lows = df["Low"].eq(df["Low"].rolling(2 * window + 1, center=True).min())
    return highs.fillna(False), lows.fillna(False)


def structure_state(df: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
    if len(df) < max(20, window * 3):
        return {"label": "Insufficient data", "bos": "Unavailable", "choch": "Unavailable"}

    high_swings, low_swings = swing_points(df, window)
    swing_highs = df.loc[high_swings, "High"]
    swing_lows = df.loc[low_swings, "Low"]

    label = "Neutral"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]
        hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]
        lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]
        ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]
        if hh and hl:
            label = "HH / HL"
        elif lh and ll:
            label = "LH / LL"
        elif hh:
            label = "Higher High"
        elif ll:
            label = "Lower Low"

    lookback = min(20, len(df) - 1)
    prior_high = float(df["High"].iloc[-lookback - 1:-1].max())
    prior_low = float(df["Low"].iloc[-lookback - 1:-1].min())
    last = float(df["Close"].iloc[-1])

    bos = "Bullish BOS" if last > prior_high else "Bearish BOS" if last < prior_low else "No BOS"
    choch = "Potential CHoCH" if (
        ("Bullish" in bos and "LL" in label) or
        ("Bearish" in bos and "HH" in label)
    ) else "No clear CHoCH"

    return {"label": label, "bos": bos, "choch": choch}


def liquidity_levels(df: pd.DataFrame, lookback: int = 30) -> Tuple[float, float]:
    n = min(lookback, len(df))
    return float(df["High"].iloc[-n:].max()), float(df["Low"].iloc[-n:].min())


def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> str:
    if len(df) < lookback + 2:
        return "Unavailable"
    prior_high = float(df["High"].iloc[-lookback - 1:-1].max())
    prior_low = float(df["Low"].iloc[-lookback - 1:-1].min())
    last = df.iloc[-1]
    if float(last["High"]) > prior_high and float(last["Close"]) < prior_high:
        return "Buy-side liquidity sweep"
    if float(last["Low"]) < prior_low and float(last["Close"]) > prior_low:
        return "Sell-side liquidity sweep"
    return "No confirmed sweep"


def detect_fvg(df: pd.DataFrame, max_items: int = 12) -> List[Dict[str, Any]]:
    gaps: List[Dict[str, Any]] = []
    for i in range(2, len(df)):
        if float(df["Low"].iloc[i]) > float(df["High"].iloc[i - 2]):
            gaps.append(
                {
                    "index": i,
                    "type": "Bullish FVG",
                    "low": float(df["High"].iloc[i - 2]),
                    "high": float(df["Low"].iloc[i]),
                }
            )
        elif float(df["High"].iloc[i]) < float(df["Low"].iloc[i - 2]):
            gaps.append(
                {
                    "index": i,
                    "type": "Bearish FVG",
                    "low": float(df["High"].iloc[i]),
                    "high": float(df["Low"].iloc[i - 2]),
                }
            )
    return gaps[-max_items:]


def detect_order_blocks(df: pd.DataFrame, max_items: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    bullish: List[Dict[str, Any]] = []
    bearish: List[Dict[str, Any]] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        if float(cur["Close"]) > float(prev["High"]):
            bullish.append(
                {
                    "index": i - 1,
                    "low": float(prev["Low"]),
                    "high": float(prev["High"]),
                    "type": "Bullish OB",
                }
            )
        elif float(cur["Close"]) < float(prev["Low"]):
            bearish.append(
                {
                    "index": i - 1,
                    "low": float(prev["Low"]),
                    "high": float(prev["High"]),
                    "type": "Bearish OB",
                }
            )
    return {"bullish": bullish[-max_items:], "bearish": bearish[-max_items:]}


def dealing_range(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    n = min(lookback, len(df))
    high = float(df["High"].iloc[-n:].max())
    low = float(df["Low"].iloc[-n:].min())
    eq = (high + low) / 2.0
    size = high - low
    return {
        "high": high,
        "low": low,
        "equilibrium": eq,
        "ote_618": high - size * 0.618,
        "ote_786": high - size * 0.786,
    }


def premium_discount(df: pd.DataFrame) -> str:
    dr = dealing_range(df)
    return "Premium" if float(df["Close"].iloc[-1]) > dr["equilibrium"] else "Discount"


def supply_demand(df: pd.DataFrame, lookback: int = 20, max_items: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    supply, demand = [], []
    if len(df) <= lookback:
        return {"supply": supply, "demand": demand}
    for i in range(lookback, len(df)):
        prior_high = float(df["High"].iloc[i - lookback:i].max())
        prior_low = float(df["Low"].iloc[i - lookback:i].min())
        if float(df["High"].iloc[i]) >= prior_high:
            supply.append({"index": i, "price": float(df["High"].iloc[i])})
        if float(df["Low"].iloc[i]) <= prior_low:
            demand.append({"index": i, "price": float(df["Low"].iloc[i])})
    return {"supply": supply[-max_items:], "demand": demand[-max_items:]}


# ---------------------------------------------------------------------------
# Session / macro context
# ---------------------------------------------------------------------------

def session_name(now_utc: Optional[datetime] = None) -> str:
    now = now_utc or datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0
    # Configurable UTC approximations. These are labels, not exchange sessions.
    london = 7 <= hour < 16
    new_york = 13 <= hour < 22
    asia = hour >= 0 and hour < 9
    if london and new_york:
        return "London / New York overlap"
    if london:
        return "London"
    if new_york:
        return "New York"
    if asia:
        return "Asia"
    return "Off-hours"


def session_windows_utc() -> Dict[str, str]:
    return {
        "London Kill Zone": "07:00–10:00 UTC (configurable approximation)",
        "New York Kill Zone": "13:00–16:00 UTC (configurable approximation)",
    }


# ---------------------------------------------------------------------------
# Deterministic confidence engine
# ---------------------------------------------------------------------------

def confidence_engine(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Controlled scoring with independent families.

    Max contributions:
      Structure ±25
      Liquidity ±15
      Zones ±15
      PD/OTE ±10
      Momentum ±15
      Volume/VWAP ±10
      Session ±5
      Macro ±5

    Macro is intentionally neutral here because this single-file app does not
    have a verified macro/news feed. The AI can be asked for current context,
    but it cannot change this deterministic technical score.
    """
    if len(df) < 60:
        return {
            "score": 0,
            "bias": "Unavailable",
            "confluences": [],
            "conflicts": ["Insufficient historical data"],
        }

    score = 50.0
    confluences: List[str] = []
    conflicts: List[str] = []

    close = float(df["Close"].iloc[-1])
    ema20 = float(ema(df["Close"], 20).iloc[-1])
    ema50 = float(ema(df["Close"], 50).iloc[-1])
    rsi_val = float(rsi(df["Close"]).iloc[-1])
    atr_val = float(atr(df).iloc[-1]) if not math.isnan(float(atr(df).iloc[-1])) else 0.0
    rv = float(relative_volume(df).iloc[-1]) if not math.isnan(float(relative_volume(df).iloc[-1])) else 1.0
    vwap_val = float(vwap(df).iloc[-1]) if not math.isnan(float(vwap(df).iloc[-1])) else close
    structure = structure_state(df)
    sweep = liquidity_sweep(df)
    dr = dealing_range(df)

    # Structure family ±25
    if "Bullish" in structure["bos"]:
        score += 25
        confluences.append("Bullish BOS")
    elif "Bearish" in structure["bos"]:
        score -= 25
        confluences.append("Bearish BOS")
    elif structure["label"] in {"HH / HL", "Higher High"}:
        score += 12
        confluences.append("Higher-high structure")
    elif structure["label"] in {"LH / LL", "Lower Low"}:
        score -= 12
        confluences.append("Lower-low structure")
    else:
        conflicts.append("No decisive structure break")

    # Liquidity family ±15
    if "buy-side" in sweep.lower():
        score -= 15
        confluences.append("Buy-side liquidity sweep")
    elif "sell-side" in sweep.lower():
        score += 15
        confluences.append("Sell-side liquidity sweep")
    else:
        conflicts.append("No confirmed liquidity sweep")

    # Zones / imbalance family ±15
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df)
    zone_bull = len([x for x in fvgs if x["type"] == "Bullish FVG"]) + len(obs["bullish"])
    zone_bear = len([x for x in fvgs if x["type"] == "Bearish FVG"]) + len(obs["bearish"])
    if zone_bull > zone_bear and zone_bull:
        score += 15
        confluences.append("Bullish imbalance/OB evidence")
    elif zone_bear > zone_bull and zone_bear:
        score -= 15
        confluences.append("Bearish imbalance/OB evidence")
    else:
        conflicts.append("Zone evidence is mixed")

    # Premium/discount/OTE ±10
    if close < dr["equilibrium"]:
        score += 10 if close >= dr["ote_786"] else 5
        confluences.append("Price in discount")
    else:
        score -= 10 if close <= dr["ote_618"] else 5
        confluences.append("Price in premium")

    # Momentum family ±15
    if close > ema20 > ema50 and rsi_val >= 55:
        score += 15
        confluences.append("Trend + RSI momentum")
    elif close < ema20 < ema50 and rsi_val <= 45:
        score -= 15
        confluences.append("Bearish trend + RSI momentum")
    else:
        conflicts.append("Momentum is not aligned")

    # Volume/VWAP family ±10
    if close > vwap_val and rv >= 1.1:
        score += 10
        confluences.append("Above VWAP with elevated relative volume")
    elif close < vwap_val and rv >= 1.1:
        score -= 10
        confluences.append("Below VWAP with elevated relative volume")
    else:
        conflicts.append("Volume/VWAP confirmation is weak")

    # Session family ±5
    sess = session_name()
    if "London" in sess or "New York" in sess:
        confluences.append(f"Active session: {sess}")
        score += 5 if close >= ema20 else -5

    # No macro contribution without verified macro data.
    conflicts.append("Macro score neutral: no verified live macro feed")

    score = int(round(max(0.0, min(100.0, score))))

    bullish_count = sum(
        1 for x in confluences
        if any(k in x.lower() for k in ["bullish", "higher", "above", "discount", "trend +"])
    )
    bearish_count = sum(
        1 for x in confluences
        if any(k in x.lower() for k in ["bearish", "lower", "below", "premium"])
    )

    if score >= 60 and bullish_count >= 3 and bullish_count >= bearish_count + 1:
        bias = "Bullish"
    elif score <= 40 and bearish_count >= 3 and bearish_count >= bullish_count + 1:
        bias = "Bearish"
    else:
        bias = "Neutral"

    if len(confluences) < 3:
        score = min(score, 59)

    return {
        "score": score,
        "bias": bias,
        "confluences": confluences,
        "conflicts": conflicts,
        "atr": atr_val,
        "rsi": rsi_val,
        "relative_volume": rv,
        "vwap": vwap_val,
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def fmt_price(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 100:
        return f"{value:,.3f}"
    if abs(value) >= 1:
        return f"{value:,.4f}"
    return f"{value:,.6f}"


def add_zone(fig: go.Figure, x0, x1, y0, y1, label: str, row: int = 1) -> None:
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line_width=1,
        fillcolor="rgba(60,150,255,0.08)",
        line_color="rgba(90,170,255,0.55)",
        row=row,
        col=1,
    )
    fig.add_annotation(
        x=x1,
        y=(y0 + y1) / 2,
        text=label,
        showarrow=False,
        font=dict(size=9),
        row=row,
        col=1,
    )


def create_chart(
    df: pd.DataFrame,
    chart_type: str,
    theme: str,
    show_ema: bool,
    show_sma: bool,
    show_bb: bool,
    show_vwap: bool,
    show_structure: bool,
    show_smc: bool,
) -> go.Figure:
    plot_template = "plotly_dark" if theme == "Dark" else "plotly_white"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.58, 0.16, 0.14, 0.12],
        subplot_titles=("PRICE", "RSI", "MACD", "VOLUME"),
    )

    if chart_type == "Heikin Ashi":
        ha = heikin_ashi(df)
        fig.add_trace(
            go.Candlestick(
                x=ha.index,
                open=ha["HA_Open"],
                high=ha["HA_High"],
                low=ha["HA_Low"],
                close=ha["HA_Close"],
                name="Heikin Ashi",
            ),
            row=1,
            col=1,
        )
    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"),
            row=1,
            col=1,
        )
    elif chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                fill="tozeroy",
                name="Close",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

    if show_ema:
        for period in (20, 50, 200):
            if len(df) >= period:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema(df["Close"], period),
                        mode="lines",
                        name=f"EMA {period}",
                        line=dict(width=1.2),
                    ),
                    row=1,
                    col=1,
                )

    if show_sma:
        for period in (20, 50, 200):
            if len(df) >= period:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma(df["Close"], period),
                        mode="lines",
                        name=f"SMA {period}",
                        line=dict(width=1, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

    if show_bb:
        upper, mid, lower = bollinger(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=mid, name="BB Mid", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower", line=dict(width=1)), row=1, col=1)

    if show_vwap:
        fig.add_trace(
            go.Scatter(x=df.index, y=vwap(df), name="VWAP", line=dict(width=1.5)),
            row=1,
            col=1,
        )

    rsi_values = rsi(df["Close"])
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi_values, name="RSI", line=dict(width=1.4)),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    macd_line, signal, hist = macd(df["Close"])
    fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD Hist", opacity=0.45), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal", line=dict(width=1.2)), row=3, col=1)

    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.7),
        row=4,
        col=1,
    )

    if show_structure:
        structure = structure_state(df)
        if "Bullish BOS" in structure["bos"]:
            fig.add_annotation(
                x=df.index[-1],
                y=df["Close"].iloc[-1],
                text="BULL BOS",
                showarrow=True,
                row=1,
                col=1,
            )
        elif "Bearish BOS" in structure["bos"]:
            fig.add_annotation(
                x=df.index[-1],
                y=df["Close"].iloc[-1],
                text="BEAR BOS",
                showarrow=True,
                row=1,
                col=1,
            )

    buy_liq, sell_liq = liquidity_levels(df)
    fig.add_hline(
        y=buy_liq,
        line_dash="dash",
        annotation_text="Buy-side liquidity",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=sell_liq,
        line_dash="dash",
        annotation_text="Sell-side liquidity",
        row=1,
        col=1,
    )

    if show_smc:
        for gap in detect_fvg(df, 6):
            idx = gap["index"]
            if 0 <= idx < len(df):
                x0 = df.index[max(0, idx - 2)]
                x1 = df.index[idx]
                add_zone(
                    fig,
                    x0,
                    x1,
                    gap["low"],
                    gap["high"],
                    gap["type"],
                    row=1,
                )

        obs = detect_order_blocks(df, 4)
        for item in obs["bullish"] + obs["bearish"]:
            idx = item["index"]
            x0 = df.index[idx]
            x1 = df.index[min(len(df) - 1, idx + 8)]
            add_zone(fig, x0, x1, item["low"], item["high"], item["type"], row=1)

        dr = dealing_range(df)
        fig.add_hline(
            y=dr["equilibrium"],
            line_dash="dot",
            annotation_text="50% EQ",
            row=1,
            col=1,
        )
        fig.add_hline(y=dr["ote_618"], line_dash="dot", annotation_text="OTE 61.8%", row=1, col=1)
        fig.add_hline(y=dr["ote_786"], line_dash="dot", annotation_text="OTE 78.6%", row=1, col=1)

    fig.update_layout(
        template=plot_template,
        height=860,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis_rangeslider_visible=False,
        uirevision=f"{st.session_state.selected_symbol}-{st.session_state.selected_tf}",
    )

    fig.update_yaxes(showgrid=True, gridcolor="rgba(130,140,160,.12)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(130,140,160,.08)")
    return fig


# ---------------------------------------------------------------------------
# AI / Groq
# ---------------------------------------------------------------------------

def get_secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value else None


def groq_client() -> Optional[OpenAI]:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
    except Exception:
        return None


def ai_model() -> str:
    return get_secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def compact_market_context(symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
    conf = confidence_engine(df)
    structure = structure_state(df)
    dr = dealing_range(df)
    fvg = detect_fvg(df, 8)
    obs = detect_order_blocks(df, 6)
    price = current_price(df)
    return {
        "symbol": symbol,
        "instrument": SYMBOLS[symbol]["name"],
        "timeframe": timeframe,
        "last_price": price,
        "percent_change_last_bar": pct_change(df),
        "rsi": conf.get("rsi"),
        "atr": conf.get("atr"),
        "relative_volume": conf.get("relative_volume"),
        "vwap": conf.get("vwap"),
        "market_structure": structure,
        "liquidity_sweep": liquidity_sweep(df),
        "liquidity": dict(zip(["buy_side", "sell_side"], liquidity_levels(df))),
        "premium_discount": premium_discount(df),
        "dealing_range": dr,
        "fvg_count": len(fvg),
        "recent_fvgs": fvg[-4:],
        "bullish_order_blocks": obs["bullish"][-3:],
        "bearish_order_blocks": obs["bearish"][-3:],
        "deterministic_confidence": conf["score"],
        "deterministic_bias": conf["bias"],
        "confluences": conf["confluences"],
        "conflicts": conf["conflicts"],
        "session": session_name(),
        "order_flow": "Unavailable. yfinance does not provide true DOM/footprint/bid-ask delta data.",
    }


def build_ai_prompt(symbol: str, timeframe: str, df: pd.DataFrame, user_question: str = "") -> str:
    context = compact_market_context(symbol, timeframe, df)
    return f"""
You are the AI analyst inside DHairya FX Pro, a PAPER-TRADING market-analysis
application. You are not a broker and must not claim to execute trades.

Use ONLY the supplied computed market context for technical claims. Do not
invent prices, indicators, news, macroeconomic releases, order flow, or
institutional activity.

Important:
- The deterministic Python confidence score is authoritative for the displayed
  technical confidence. Do not replace or invent a different score.
- SMC/ICT labels are algorithmic approximations.
- yfinance is not a professional exchange feed and may be delayed.
- True order flow is unavailable.
- If current macro/news is not supplied, say "Current news/macro: Data unavailable"
  rather than guessing.
- This is not financial advice.

Computed context:
{json.dumps(context, default=str, indent=2)}

User request:
{user_question or "Give a concise professional market analysis."}

Return these sections:
BIAS
CONFIDENCE (repeat the supplied deterministic score only)
CONFLUENCES
CONFLICTS
ENTRY ZONE (only as an educational hypothetical zone; do not present as a guaranteed trade)
INVALIDATION
TARGET 1
TARGET 2
SMC / ICT REASONING
MACRO / NEWS
WHAT WOULD INVALIDATE THE THESIS
DISCLAIMER

If current news is required but not supplied, explicitly state that it is
unavailable. Never fabricate current events.
"""


@st.cache_data(ttl=120, show_spinner=False)
def run_ai_analysis(symbol: str, timeframe: str, df: pd.DataFrame, user_question: str) -> str:
    client = groq_client()
    if client is None:
        return (
            "AI is offline because GROQ_API_KEY is not configured. "
            "Add it to Streamlit secrets or the environment. No key is embedded "
            "in this application source."
        )

    prompt = build_ai_prompt(symbol, timeframe, df, user_question)

    try:
        response = client.chat.completions.create(
            model=ai_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful quantitative market-analysis assistant. "
                        "Never fabricate current information. Distinguish computed "
                        "technical data from uncertainty."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            max_tokens=1800,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "AI returned no text."
    except Exception as exc:
        return f"AI request failed safely: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=120, show_spinner=False)
def run_ai_chat(symbol: str, timeframe: str, df: pd.DataFrame, question: str) -> str:
    return run_ai_analysis(symbol, timeframe, df, question)


# ---------------------------------------------------------------------------
# Paper-trading simulation
# ---------------------------------------------------------------------------

def simulate_open_positions() -> int:
    trades = get_trades()
    if trades.empty:
        return 0

    closed_count = 0
    for _, trade in trades[trades["status"] == "OPEN"].iterrows():
        data = get_market_data(str(trade["symbol"]), st.session_state.selected_tf)
        if data.empty:
            continue

        # Use the newest bar only. If both SL and TP are touched inside one
        # candle, use a deterministic conservative rule: SL is checked first.
        bar = data.iloc[-1]
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        side = str(trade["side"])
        sl = trade["stop_loss"]
        tp = trade["take_profit"]

        reason = None
        fill = close

        if side == "BUY":
            if pd.notna(sl) and low <= float(sl):
                reason, fill = "SL hit", float(sl)
            elif pd.notna(tp) and high >= float(tp):
                reason, fill = "TP hit", float(tp)
        else:
            if pd.notna(sl) and high >= float(sl):
                reason, fill = "SL hit", float(sl)
            elif pd.notna(tp) and low <= float(tp):
                reason, fill = "TP hit", float(tp)

        if reason:
            if close_trade(int(trade["id"]), fill, reason):
                closed_count += 1

    return closed_count


def performance_stats(trades: pd.DataFrame, starting_balance: float) -> Dict[str, float]:
    if trades.empty:
        return {
            "total": 0,
            "open": 0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "max_dd": 0.0,
            "balance": starting_balance,
        }

    closed = trades[trades["status"] == "CLOSED"].copy()
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] < 0]

    pnl = float(closed["pnl"].sum())
    win_rate = (len(wins) / len(closed) * 100.0) if len(closed) else 0.0
    gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 0.0
    profit_factor = gross_win / gross_loss if gross_loss else (math.inf if gross_win else 0.0)

    rs = []
    for _, row in closed.iterrows():
        stop_distance = abs(float(row["entry"]) - float(row["stop_loss"])) if pd.notna(row["stop_loss"]) else 0.0
        risk_money = stop_distance * float(row["quantity"])
        if risk_money > 0:
            rs.append(float(row["pnl"]) / risk_money)

    equity = starting_balance + closed.sort_values("id")["pnl"].cumsum()
    if len(equity):
        peak = equity.cummax()
        drawdown = equity - peak
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    return {
        "total": len(trades),
        "open": int((trades["status"] == "OPEN").sum()),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "pnl": pnl,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "avg_r": float(np.mean(rs)) if rs else 0.0,
        "max_dd": max_dd,
        "balance": starting_balance + pnl,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_symbol_from_text(text: str) -> Optional[str]:
    normalized = text.upper().strip()
    # Longest aliases first prevents "XAU" from being checked after "XAUUSD".
    for alias in sorted(ALIASES.keys(), key=len, reverse=True):
        if alias in normalized:
            return ALIASES[alias]
    return None


def detect_timeframe_from_text(text: str) -> Optional[str]:
    normalized = text.lower()
    patterns = [
        ("1m", r"\b1\s*m(in)?\b"),
        ("5m", r"\b5\s*m(in)?\b"),
        ("15m", r"\b15\s*m(in)?\b"),
        ("1H", r"\b1\s*h(our)?\b"),
        ("4H", r"\b4\s*h(our)?\b"),
        ("1D", r"\b1\s*d(ay)?\b"),
        ("1W", r"\b1\s*w(eek)?\b"),
    ]
    for tf, pattern in patterns:
        if re.search(pattern, normalized):
            return tf
    return None


def position_size(account_size: float, risk_pct: float, entry: float, stop: float) -> Tuple[float, float]:
    risk_amount = max(0.0, account_size) * max(0.0, risk_pct) / 100.0
    distance = abs(entry - stop)
    qty = risk_amount / distance if distance > 0 else 0.0
    return risk_amount, qty


def save_note(note: str) -> None:
    if not note.strip():
        return
    conn = db_connect()
    conn.execute(
        "INSERT INTO notes(note, created_at) VALUES(?, ?)",
        (note.strip(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 DHairya FX Pro")
    st.caption("Professional analysis • paper trading only")

    st.selectbox(
        "Theme",
        ["Dark", "Light"],
        key="theme",
    )

    categories = ["All", "Forex", "Metals", "Energy", "Crypto", "Indices"]
    category = st.selectbox("Market", categories)

    available_symbols = [
        s for s, meta in SYMBOLS.items()
        if category == "All" or meta["category"] == category
    ]

    current_index = available_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in available_symbols else 0
    selected = st.selectbox(
        "Instrument",
        available_symbols,
        index=current_index,
        format_func=lambda x: f"{x} — {SYMBOLS[x]['name']}",
    )
    st.session_state.selected_symbol = selected

    tf = st.selectbox(
        "Timeframe",
        list(TIMEFRAMES.keys()),
        index=list(TIMEFRAMES.keys()).index(st.session_state.selected_tf),
    )
    st.session_state.selected_tf = tf

    st.selectbox(
        "Chart",
        ["Candlestick", "Heikin Ashi", "Line", "Area"],
        key="chart_type",
    )

    st.divider()
    st.markdown("### Watchlist")

    watch_rows = []
    for symbol in available_symbols:
        wd = get_watch_data(symbol)
        if wd.empty:
            watch_rows.append({"Symbol": symbol, "Price": "Data unavailable", "Change": "—"})
            continue
        price = current_price(wd)
        change = pct_change(wd)
        watch_rows.append(
            {
                "Symbol": symbol,
                "Price": fmt_price(price),
                "Change": f"{change:+.2f}%" if np.isfinite(change) else "—",
            }
        )

    if watch_rows:
        st.dataframe(
            pd.DataFrame(watch_rows),
            use_container_width=True,
            hide_index=True,
            height=310,
        )

    st.divider()
    st.markdown("### Account & Risk")
    st.number_input("Paper account ($)", min_value=0.0, step=500.0, key="account_size")
    st.number_input("Risk per trade (%)", min_value=0.01, max_value=20.0, step=0.25, key="risk_pct")

    st.divider()
    refresh = st.button("↻ Refresh market data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.caption("Yahoo Finance data may be delayed and limited.")
    st.caption(f"AI: {'Online' if groq_client() else 'Offline'}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

if refresh:
    st.cache_data.clear()

symbol = st.session_state.selected_symbol
timeframe = st.session_state.selected_tf

with st.spinner(f"Loading {symbol} · {timeframe}…"):
    market_df = get_market_data(symbol, timeframe)

if market_df.empty:
    st.error(
        "Data unavailable. Yahoo Finance did not return usable OHLCV data for "
        f"{SYMBOLS[symbol]['name']} on {timeframe}."
    )
    st.stop()

# Keep chart responsive.
chart_df = market_df.tail(1000).copy()

# Simulate SL/TP before displaying the latest portfolio state.
simulate_open_positions()

price = current_price(chart_df)
change = pct_change(chart_df)
conf = confidence_engine(chart_df)
structure = structure_state(chart_df)
dr = dealing_range(chart_df)
sweep = liquidity_sweep(chart_df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="fx-card">
        <div class="fx-title">DHairya FX Pro</div>
        <div class="fx-subtitle">Market analysis terminal · {SYMBOLS[symbol]['name']} · {timeframe}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Last", fmt_price(price), f"{change:+.2f}%" if np.isfinite(change) else None)
m2.metric("Bias", conf["bias"])
m3.metric("Confidence", f"{conf['score']}/100")
m4.metric("RSI", f"{conf['rsi']:.1f}" if np.isfinite(conf["rsi"]) else "—")
m5.metric("Relative Vol.", f"{conf['relative_volume']:.2f}x" if np.isfinite(conf["relative_volume"]) else "—")
m6.metric("Session", session_name())

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
with c1:
    show_ema = st.checkbox("EMA 20/50/200", True)
with c2:
    show_sma = st.checkbox("SMA 20/50/200", False)
with c3:
    show_bb = st.checkbox("Bollinger", False)
with c4:
    show_vwap = st.checkbox("VWAP", True)
with c5:
    show_structure = st.checkbox("Structure", True)
with c6:
    show_smc = st.checkbox("SMC/ICT", True)

# ---------------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------------

left, right = st.columns([2.65, 1], gap="large")

with left:
    chart = create_chart(
        chart_df,
        st.session_state.chart_type,
        st.session_state.theme,
        show_ema,
        show_sma,
        show_bb,
        show_vwap,
        show_structure,
        show_smc,
    )
    st.plotly_chart(
        chart,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawrect",
                "drawcircle",
                "eraseshape",
            ],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.caption(
        "Plotly drawing tools are session/browser-side. Persistent arbitrary mouse "
        "drawings are not guaranteed by Streamlit."
    )

with right:
    st.markdown("### AI Analyst")
    ai_status = "Connected" if groq_client() else "Offline"
    st.markdown(
        f'<span class="badge badge-blue">GROQ · {ai_status}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Technical Thesis")
    st.write(f"**Bias:** {conf['bias']}")
    st.write(f"**Deterministic confidence:** {conf['score']}/100")
    st.write(f"**Structure:** {structure['label']}")
    st.write(f"**BOS:** {structure['bos']}")
    st.write(f"**CHoCH:** {structure['choch']}")
    st.write(f"**Liquidity:** {sweep}")
    st.write(f"**PD Array:** {premium_discount(chart_df)}")

    st.markdown("#### Confluences")
    for item in conf["confluences"][:7]:
        st.write(f"• {item}")

    st.markdown("#### Conflicts")
    for item in conf["conflicts"][:5]:
        st.write(f"• {item}")

    if st.button("Run AI market analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing computed market context…"):
            st.session_state.ai_result = run_ai_analysis(
                symbol,
                timeframe,
                chart_df,
                "Analyze the current market using the computed context.",
            )

    if st.session_state.ai_result:
        st.markdown(st.session_state.ai_result)


# ---------------------------------------------------------------------------
# Paper trading
# ---------------------------------------------------------------------------

st.markdown("## Paper Trading")

trade_col, risk_col = st.columns([1.3, 1])

with trade_col:
    st.markdown("### Trading Ticket")
    side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
    entry = st.number_input("Entry", value=float(price), min_value=0.0, format="%.8f")
    stop_default = entry - conf["atr"] if side == "BUY" and np.isfinite(conf["atr"]) and conf["atr"] > 0 else entry * (0.99 if side == "BUY" else 1.01)
    tp_default = entry + 2 * abs(entry - stop_default) if side == "BUY" else entry - 2 * abs(entry - stop_default)
    stop = st.number_input("Stop Loss", value=float(max(0.0, stop_default)), min_value=0.0, format="%.8f")
    take_profit = st.number_input("Take Profit", value=float(max(0.0, tp_default)), min_value=0.0, format="%.8f")

    risk_money, suggested_qty = position_size(
        float(st.session_state.account_size),
        float(st.session_state.risk_pct),
        entry,
        stop,
    )
    quantity = st.number_input(
        "Quantity",
        min_value=0.000001,
        value=float(max(suggested_qty, 0.000001)),
        format="%.6f",
    )

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Risk $", f"{risk_money:,.2f}")
    rc2.metric("Stop distance", fmt_price(abs(entry - stop)))
    rc3.metric("Suggested qty", f"{suggested_qty:.4f}")

    if st.button("Open paper position", type="primary", use_container_width=True):
        valid = True
        if entry <= 0 or quantity <= 0:
            valid = False
            st.error("Entry and quantity must be greater than zero.")
        if side == "BUY" and not (stop < entry < take_profit):
            valid = False
            st.error("For BUY, require Stop < Entry < Take Profit.")
        if side == "SELL" and not (take_profit < entry < stop):
            valid = False
            st.error("For SELL, require Take Profit < Entry < Stop.")

        if valid:
            open_paper_trade(symbol, side, quantity, entry, stop, take_profit)
            st.success("Paper position opened.")
            st.rerun()

with risk_col:
    st.markdown("### Risk Model")
    st.info(
        f"""
        **Account:** ${st.session_state.account_size:,.2f}

        **Risk:** {st.session_state.risk_pct:.2f}%

        **Risk amount:** ${risk_money:,.2f}

        **Position size:** {suggested_qty:.4f}

        Position sizing is simplified as risk amount ÷ price distance.
        Contract specifications, pip values, leverage, fees and currency
        conversion can materially change real-world sizing.
        """
    )

# ---------------------------------------------------------------------------
# Positions / history
# ---------------------------------------------------------------------------

trades = get_trades()
stats = performance_stats(trades, float(st.session_state.account_size))

st.markdown("## Portfolio")

p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("Balance", f"${stats['balance']:,.2f}")
p2.metric("Total Trades", stats["total"])
p3.metric("Open", stats["open"])
p4.metric("Win Rate", f"{stats['win_rate']:.1f}%")
p5.metric("Profit Factor", "∞" if math.isinf(stats["profit_factor"]) else f"{stats['profit_factor']:.2f}")
p6.metric("Max Drawdown", f"${stats['max_dd']:,.2f}")

if not trades.empty:
    st.dataframe(
        trades[
            [
                "id",
                "symbol",
                "side",
                "quantity",
                "entry",
                "stop_loss",
                "take_profit",
                "status",
                "pnl",
                "opened_at",
                "closed_at",
                "close_reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    open_ids = trades.loc[trades["status"] == "OPEN", "id"].tolist()
    if open_ids:
        close_col1, close_col2 = st.columns([1, 2])
        with close_col1:
            selected_trade_id = st.selectbox("Open position ID", open_ids)
        with close_col2:
            if st.button("Close selected position at latest market price", use_container_width=True):
                close_trade(int(selected_trade_id), price, "Manual close")
                st.success("Paper position closed.")
                st.rerun()

    csv_bytes = trades.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export trade history CSV",
        data=csv_bytes,
        file_name="dhairyafxpro_trades.csv",
        mime="text/csv",
    )
else:
    st.info("No paper trades yet.")

# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

st.markdown("## Performance Analytics")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Closed", stats["closed"])
a2.metric("Winning", stats["wins"])
a3.metric("Losing", stats["losses"])
a4.metric("Average R", f"{stats['avg_r']:.2f}")

if not trades.empty:
    closed = trades[trades["status"] == "CLOSED"].sort_values("id").copy()
    if not closed.empty:
        closed["Equity"] = float(st.session_state.account_size) + closed["pnl"].cumsum()
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=closed["id"], y=closed["Equity"], mode="lines+markers", name="Equity"))
        eq_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            template="plotly_dark" if st.session_state.theme == "Dark" else "plotly_white",
            xaxis_title="Trade ID",
            yaxis_title="Account equity",
        )
        st.plotly_chart(eq_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Market structure / SMC dashboard
# ---------------------------------------------------------------------------

st.markdown("## Structure & SMC Dashboard")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Structure", structure["label"])
s2.metric("BOS", structure["bos"])
s3.metric("CHoCH", structure["choch"])
s4.metric("Liquidity sweep", sweep)

smc1, smc2, smc3, smc4 = st.columns(4)
fvg_items = detect_fvg(chart_df)
obs_items = detect_order_blocks(chart_df)
sd = supply_demand(chart_df)
smc1.metric("FVGs", len(fvg_items))
smc2.metric("Bullish OBs", len(obs_items["bullish"]))
smc3.metric("Bearish OBs", len(obs_items["bearish"]))
smc4.metric("Supply/Demand", f"{len(sd['supply'])}/{len(sd['demand'])}")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Range high", fmt_price(dr["high"]))
d2.metric("Equilibrium", fmt_price(dr["equilibrium"]))
d3.metric("OTE 61.8%", fmt_price(dr["ote_618"]))
d4.metric("OTE 78.6%", fmt_price(dr["ote_786"]))

st.caption(
    "SMC/ICT, BOS/CHoCH, order blocks, FVGs, liquidity and supply/demand are "
    "algorithmic approximations. They are not guaranteed institutional classifications."
)

# ---------------------------------------------------------------------------
# AI chat
# ---------------------------------------------------------------------------

st.markdown("## AI Chat")
st.caption("Examples: “Analyze Gold on 15m”, “What is the current structure on GBPUSD?”, “Show BTC on 1H”.")

chat_input = st.chat_input("Ask DHairya FX Pro…")

if chat_input:
    detected_symbol = detect_symbol_from_text(chat_input)
    detected_tf = detect_timeframe_from_text(chat_input)

    if detected_symbol:
        st.session_state.selected_symbol = detected_symbol
        symbol = detected_symbol

    if detected_tf:
        st.session_state.selected_tf = detected_tf
        timeframe = detected_tf

    if detected_symbol or detected_tf:
        st.rerun()

    st.session_state.chat_history.append(("user", chat_input))
    with st.spinner("Thinking from the computed market context…"):
        answer = run_ai_chat(symbol, timeframe, chart_df, chat_input)
    st.session_state.chat_history.append(("assistant", answer))

for role, message in st.session_state.chat_history[-12:]:
    with st.chat_message(role):
        st.markdown(message)

# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

st.markdown("## Trade Journal")
journal = st.text_area("Private note", placeholder="Record setup, execution, emotions and lessons…")
if st.button("Save journal note"):
    save_note(journal)
    st.success("Journal note saved.")

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

st.markdown("## Market Scanner")
scanner_rows = []
for scan_symbol in SYMBOLS:
    scan_df = get_market_data(scan_symbol, "1H")
    if scan_df.empty or len(scan_df) < 60:
        continue
    scan_conf = confidence_engine(scan_df)
    scanner_rows.append(
        {
            "Symbol": scan_symbol,
            "Instrument": SYMBOLS[scan_symbol]["name"],
            "Price": fmt_price(current_price(scan_df)),
            "Change": f"{pct_change(scan_df):+.2f}%",
            "RSI": round(float(rsi(scan_df["Close"]).iloc[-1]), 1),
            "Bias": scan_conf["bias"],
            "Confidence": scan_conf["score"],
        }
    )

if scanner_rows:
    scanner_df = pd.DataFrame(scanner_rows).sort_values("Confidence", ascending=False)
    st.dataframe(scanner_df, use_container_width=True, hide_index=True)
else:
    st.info("Scanner data unavailable.")

# ---------------------------------------------------------------------------
# Command helper
# ---------------------------------------------------------------------------

with st.expander("Command / chart control"):
    command = st.text_input(
        "Type a chart command",
        placeholder="Show Gold on 15m",
        value=st.session_state.last_command,
    )
    if st.button("Apply command"):
        st.session_state.last_command = command
        cmd_symbol = detect_symbol_from_text(command)
        cmd_tf = detect_timeframe_from_text(command)
        if cmd_symbol:
            st.session_state.selected_symbol = cmd_symbol
        if cmd_tf:
            st.session_state.selected_tf = cmd_tf
        if cmd_symbol or cmd_tf:
            st.success(
                f"Chart command applied: "
                f"{st.session_state.selected_symbol} · {st.session_state.selected_tf}"
            )
            st.rerun()
        else:
            st.warning("No supported symbol/timeframe was detected.")

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

st.markdown("---")
st.warning(
    """
**Paper Trading & Data Disclaimer**

• This application never executes real broker orders.
• All positions and fills are simulated.
• Yahoo Finance data can be delayed, incomplete or unavailable; it is not a professional exchange feed.
• Intraday historical availability is limited by Yahoo Finance.
• True institutional order flow, DOM, Level 2, footprint and bid/ask delta are unavailable.
• SMC/ICT/BOS/CHoCH/OB/FVG/OTE detection is algorithmic and approximate.
• AI output can be incorrect and should not be treated as financial advice.
• Current macro/news is marked unavailable unless verified data is supplied.
• When an OHLC candle touches both SL and TP, the simulator uses a deterministic
  conservative rule: SL is checked first; exact intrabar execution order is unknown.
"""
)

st.caption(
    f"{APP_NAME} · Paper Trading Only · Yahoo Finance + Groq AI · "
    f"Model: {ai_model()}"
)
