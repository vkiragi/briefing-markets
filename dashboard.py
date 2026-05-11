"""
Briefing Markets — price chart, key stats, news, AI briefing, and AI technical
analysis.

Run with:
    pip install -r requirements.txt
    streamlit run dashboard.py

Data sources:
    - Quote, profile, fundamentals, news -> Finnhub (FINNHUB_API_KEY)
    - Historical bars (chart)            -> Alpaca IEX (ALPACA_API_KEY/SECRET)
AI:
    - OpenRouter -> Claude Haiku 4.5 (OPENROUTER_API_KEY)
"""

from datetime import date, datetime, timedelta, timezone
import os

from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FINNHUB_API_KEY    = os.getenv("FINNHUB_API_KEY")
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET  = os.getenv("ALPACA_API_SECRET")

OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"
PRESET_TICKERS   = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN", "META"]

# How many trading days to fetch per chart range. Pad so MAs have warmup.
PERIOD_DAYS = {
    "1mo": 60, "3mo": 120, "6mo": 200, "1y": 365, "2y": 730, "5y": 1825,
}

st.set_page_config(page_title="Briefing Markets", page_icon="📈", layout="wide")


# ============================================================
# Data fetching
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_finnhub(ticker: str):
    """Returns (quote, profile, metrics, news, errors). Quote/profile/metrics
    are dicts; news is a list."""
    errors = {}
    base = "https://finnhub.io/api/v1"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}

    def _get(path, extra=None):
        try:
            r = requests.get(f"{base}/{path}", params={**params, **(extra or {})}, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            errors[path] = f"{type(e).__name__}: {e}"
            return {}

    quote   = _get("quote")
    profile = _get("stock/profile2")
    metrics = (_get("stock/metric", {"metric": "all"}) or {}).get("metric", {})

    to_d = date.today().isoformat()
    fr_d = (date.today() - timedelta(days=14)).isoformat()
    news_raw = _get("company-news", {"from": fr_d, "to": to_d})
    news = news_raw if isinstance(news_raw, list) else []

    return quote, profile, metrics, news, errors


@st.cache_data(ttl=300, show_spinner=False)
def fetch_bars(ticker: str, days: int) -> pd.DataFrame:
    """Daily bars from Alpaca IEX feed. Returns DataFrame indexed by date with
    columns: open, high, low, close, volume."""
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        return pd.DataFrame()
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )).df
    except Exception:
        return pd.DataFrame()
    if bars.empty:
        return bars
    if "symbol" in bars.index.names:
        bars = bars.xs(ticker, level="symbol")
    bars.index = pd.to_datetime(bars.index).tz_convert(None).normalize()
    return bars[["open", "high", "low", "close", "volume"]]


# ============================================================
# Indicators
# ============================================================

def compute_indicators(bars: pd.DataFrame) -> dict:
    """Compute MAs, RSI, MACD, Bollinger Bands, volume ratio. Returns a dict
    with the full series (for plotting) and the latest values (for the prompt)."""
    if bars.empty or len(bars) < 20:
        return {}

    close = bars["close"]
    vol   = bars["volume"]

    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # RSI(14) — Wilder's smoothing
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-10)
    rsi   = 100 - 100 / (1 + rs)

    # MACD: 12-EMA - 26-EMA, signal = 9-EMA of MACD
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist_h = macd - signal

    # Bollinger Bands (20, 2σ)
    bb_mid = ma20
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std

    vol_avg = vol.rolling(20).mean()
    vol_ratio = vol.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] else None

    last_close = float(close.iloc[-1])

    def _f(s):
        v = s.iloc[-1]
        return float(v) if pd.notna(v) else None

    return {
        "series": {
            "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "rsi": rsi, "macd": macd, "signal": signal, "hist": hist_h,
            "bb_up": bb_up, "bb_dn": bb_dn, "bb_mid": bb_mid,
        },
        "latest": {
            "close":   last_close,
            "ma20":    _f(ma20),
            "ma50":    _f(ma50),
            "ma200":   _f(ma200),
            "rsi":     _f(rsi),
            "macd":    _f(macd),
            "signal":  _f(signal),
            "bb_up":   _f(bb_up),
            "bb_dn":   _f(bb_dn),
            "vol_ratio": float(vol_ratio) if vol_ratio else None,
            "above_200ma": (last_close > _f(ma200)) if _f(ma200) else None,
            "above_50ma":  (last_close > _f(ma50))  if _f(ma50)  else None,
            "above_20ma":  (last_close > _f(ma20))  if _f(ma20)  else None,
        },
    }


# ============================================================
# Formatting helpers
# ============================================================

def fmt_big(n):
    if n is None or not isinstance(n, (int, float)):
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.2f}"


def fmt_money(n):
    if n is None or not isinstance(n, (int, float)):
        return "—"
    return f"${n:,.2f}"


def fmt_pct(n, digits=2):
    if n is None or not isinstance(n, (int, float)):
        return "—"
    return f"{n:+.{digits}f}%"


# ============================================================
# OpenRouter / Claude
# ============================================================

def call_claude(prompt: str, max_tokens: int = 600) -> str:
    return call_claude_messages(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )


def call_claude_messages(messages: list, max_tokens: int = 600,
                         system: str | None = None) -> str:
    """Lower-level helper that takes a full message list (for chat). Optional
    system prompt is prepended."""
    if not OPENROUTER_API_KEY:
        return "⚠️ No OPENROUTER_API_KEY in .env."
    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": OPENROUTER_MODEL,
                "messages": payload_messages,
                "max_tokens": max_tokens,
            },
            timeout=45,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ AI call failed: {e}"


def build_chat_context(ticker: str, profile: dict, quote: dict, metrics: dict,
                       ind: dict, news: list,
                       briefing: str | None, ta: str | None) -> str:
    """One big system prompt: everything Claude needs to answer questions
    about what's currently on the page."""
    L = (ind or {}).get("latest", {})
    headlines = "\n".join(f"- {n.get('headline')}" for n in news[:5] if n.get("headline")) or "(none)"

    def _g(d, k, fmt=lambda v: v):
        v = d.get(k) if d else None
        try: return fmt(v) if v is not None else "n/a"
        except Exception: return "n/a"

    parts = [
        f"You are a stock market assistant embedded in a dashboard the user is looking at.",
        f"Answer questions about {ticker} ({profile.get('name', ticker)}) using ONLY the data below.",
        f"If asked about a different ticker or something not in this data, say you only have context for {ticker} on this page.",
        f"Be concise (2-4 short paragraphs max). Use plain English. Define jargon when you use it.",
        f"End with 'Not financial advice.' if you make any directional call.",
        "",
        "=== CURRENT SNAPSHOT ===",
        f"Ticker:       {ticker}",
        f"Name:         {profile.get('name', 'n/a')}",
        f"Sector:       {profile.get('finnhubIndustry', 'n/a')}",
        f"Price:        ${_g(quote, 'c', lambda v: f'{v:.2f}')}",
        f"Day change:   {_g(quote, 'd', lambda v: f'{v:+.2f}')} ({_g(quote, 'dp', lambda v: f'{v:+.2f}%')})",
        f"Market cap:   {_g(profile, 'marketCapitalization', lambda v: fmt_big(v * 1e6))}",
        f"P/E (TTM):    {_g(metrics, 'peTTM', lambda v: f'{v:.1f}')}",
        f"52w high:     ${_g(metrics, '52WeekHigh', lambda v: f'{v:.2f}')}",
        f"52w low:      ${_g(metrics, '52WeekLow', lambda v: f'{v:.2f}')}",
        f"EPS (TTM):    ${_g(metrics, 'epsTTM', lambda v: f'{v:.2f}')}",
        f"Beta:         {_g(metrics, 'beta', lambda v: f'{v:.2f}')}",
        "",
        "=== TECHNICAL INDICATORS ===",
        f"20-day MA:    ${_g(L, 'ma20',  lambda v: f'{v:.2f}')}  (price is {'above' if L.get('above_20ma')  else 'below'})",
        f"50-day MA:    ${_g(L, 'ma50',  lambda v: f'{v:.2f}')}  (price is {'above' if L.get('above_50ma')  else 'below'})",
        f"200-day MA:   ${_g(L, 'ma200', lambda v: f'{v:.2f}')}  (price is {'above' if L.get('above_200ma') else 'below'})",
        f"RSI(14):      {_g(L, 'rsi',     lambda v: f'{v:.1f}')}",
        f"MACD:         {_g(L, 'macd',    lambda v: f'{v:.3f}')}  (signal: {_g(L, 'signal', lambda v: f'{v:.3f}')})",
        f"Bollinger:    upper ${_g(L, 'bb_up', lambda v: f'{v:.2f}')}  /  lower ${_g(L, 'bb_dn', lambda v: f'{v:.2f}')}",
        f"Vol vs 20d:   {_g(L, 'vol_ratio', lambda v: f'{v:.2f}x')}",
        "",
        "=== RECENT HEADLINES ===",
        headlines,
    ]
    if briefing:
        parts += ["", "=== AI NEWS BRIEFING (already shown to user) ===", briefing]
    if ta:
        parts += ["", "=== AI TECHNICAL ANALYSIS (already shown to user) ===", ta]
    return "\n".join(parts)


def ai_news_briefing(ticker: str, profile: dict, quote: dict, news: list) -> str:
    headlines = [f"- {n.get('headline')}" for n in news[:8] if n.get("headline")]
    headlines_block = "\n".join(headlines) or "(no recent headlines)"
    prompt = f"""You are a concise financial analyst. Write exactly 3 short bullet
points summarizing what's going on with {ticker} ({profile.get('name', ticker)}).
Cover: (1) recent price action, (2) the dominant news theme, (3) one thing to watch.
No disclaimers, no fluff.

Current price: {fmt_money(quote.get('c'))}
Day change:    {fmt_pct(quote.get('dp'))}
Sector:        {profile.get('finnhubIndustry', 'n/a')}

Recent headlines:
{headlines_block}
"""
    return call_claude(prompt, max_tokens=400)


def ai_technical_analysis(ticker: str, profile: dict, ind: dict) -> str:
    """Beginner-mode technical analysis. Hand Claude the actual numbers and ask
    for plain-English reasoning, not buzzwords."""
    if not ind:
        return "Not enough price history to compute indicators (need 20+ trading days)."

    L = ind["latest"]
    rsi_label = (
        "OVERBOUGHT (>70)" if L["rsi"] and L["rsi"] > 70
        else "OVERSOLD (<30)" if L["rsi"] and L["rsi"] < 30
        else "NEUTRAL (30-70)"
    )
    macd_cross = (
        "BULLISH (MACD above signal)" if L["macd"] and L["signal"] and L["macd"] > L["signal"]
        else "BEARISH (MACD below signal)"
    )
    trend_bits = []
    if L["above_20ma"]  is not None: trend_bits.append(f"{'above' if L['above_20ma']  else 'below'} 20-day MA")
    if L["above_50ma"]  is not None: trend_bits.append(f"{'above' if L['above_50ma']  else 'below'} 50-day MA")
    if L["above_200ma"] is not None: trend_bits.append(f"{'above' if L['above_200ma'] else 'below'} 200-day MA")

    # Pre-format values that might be None — easier than nesting conditionals in the f-string
    def _money(v): return f"${v:.2f}" if v is not None else "n/a"
    def _num(v, d=3): return f"{v:.{d}f}" if v is not None else "n/a"
    above20  = "above" if L["above_20ma"]  else "below" if L["above_20ma"]  is not None else "n/a"

    prompt = f"""You are explaining technical analysis to someone learning. They are
looking at {ticker} ({profile.get('name', ticker)}). Use the REAL indicator values
below — do not invent numbers. Do NOT add a title or H1/H2 heading at the top —
start directly with the first ### section. Write in three sections with these
exact headers:

### What the indicators say
Walk through each indicator in plain English. For each, state the value, what
it means in this context, and WHY that matters. Avoid jargon; if you must use
a term, define it in the same sentence.

### Where signals agree or conflict
This is the most useful part. Trend, momentum, and volatility signals often
disagree. Call out where they line up and where they don't. Be specific.

### What to watch next
Give 2-3 concrete price levels or conditions that would change the read. Format:
"If [X happens], that would suggest [Y]." Keep it actionable.

INDICATOR DATA (current values):
- Price:                {_money(L['close'])}
- 20-day MA:            {_money(L['ma20'])}      ({above20})
- 50-day MA:            {_money(L['ma50'])}
- 200-day MA:           {_money(L['ma200'])}
- Trend position:       {', '.join(trend_bits) or 'n/a'}
- RSI(14):              {_num(L['rsi'], 1)}        [{rsi_label}]
- MACD:                 {_num(L['macd'])}       (signal: {_num(L['signal'])})  [{macd_cross}]
- Bollinger upper:      {_money(L['bb_up'])}
- Bollinger lower:      {_money(L['bb_dn'])}
- Volume vs 20-day avg: {_num(L['vol_ratio'], 2)}x today

Be honest. If signals are mixed, say so. If you don't have enough info for a
confident read, say that too. End with the line: "Not financial advice."
"""
    return call_claude(prompt, max_tokens=900)


# ============================================================
# Portfolio: Schwab CSV parser
# ============================================================

import io as _io
import re as _re
from pathlib import Path

# Persistent portfolio store — auto-saved on upload, auto-loaded on startup.
# Each account is a CSV under ~/.stock_dashboard/portfolios/<account_slug>.csv
PORTFOLIO_DIR      = Path.home() / ".stock_dashboard"
PORTFOLIOS_DIR     = PORTFOLIO_DIR / "portfolios"
LEGACY_SINGLE_FILE = PORTFOLIO_DIR / "portfolio.csv"   # pre-multi-account file
PROFILE_FILE       = PORTFOLIO_DIR / "profile.json"
SETTINGS_FILE      = PORTFOLIO_DIR / "settings.json"

# Default UI labels (overridable via settings.json — see load_settings)
DEFAULT_OWNER_LABELS = {
    "mine": "My Portfolio",
    "moms": "Mom's Portfolio",
}
COMBINED_KEY       = "__combined__"  # sentinel for the aggregated view


def _slug(name: str) -> str:
    """Convert an account name to a safe filename."""
    s = _re.sub(r"[^A-Za-z0-9 _-]+", "", (name or "").strip()).strip()
    s = _re.sub(r"\s+", "_", s)
    return s or "account"


def list_account_files() -> dict[str, Path]:
    """Return {display_name: path} for every saved portfolio CSV."""
    PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
    # Migrate legacy single-portfolio file on first run
    if LEGACY_SINGLE_FILE.exists() and not any(PORTFOLIOS_DIR.iterdir()):
        target = PORTFOLIOS_DIR / "Individual.csv"
        target.write_bytes(LEGACY_SINGLE_FILE.read_bytes())
        LEGACY_SINGLE_FILE.unlink()
    out = {}
    for p in sorted(PORTFOLIOS_DIR.glob("*.csv")):
        # filename stem with underscores → spaces for display
        out[p.stem.replace("_", " ")] = p
    return out


def combine_portfolios(per_account: dict) -> tuple[pd.DataFrame, dict, float]:
    """Merge multiple parsed portfolios into a single aggregate view.
    `per_account` is {name: {'holdings': df, 'totals': dict, 'cash': float}}.
    Returns (holdings_df, totals, cash) suitable for the existing UI code."""
    all_holdings, total_cash = [], 0.0
    for name, p in per_account.items():
        h = p["holdings"].copy()
        h["__account"] = name
        all_holdings.append(h)
        total_cash += p.get("cash") or 0.0

    if not all_holdings:
        return pd.DataFrame(), {}, 0.0

    merged = pd.concat(all_holdings, ignore_index=True)

    # Group by symbol — sum shares, market_value, cost_basis, gain_$, day_change_$
    # Recompute % columns from the summed values so they're internally consistent.
    grouped = merged.groupby("symbol", as_index=False).agg({
        "name":         "first",
        "shares":       "sum",
        "market_value": "sum",
        "cost_basis":   "sum",
        "gain_$":       "sum",
        "day_change_$": "sum",
        "asset_type":   "first",
    })
    # Recompute derived percentages
    total_mv = grouped["market_value"].sum() + total_cash
    grouped["price"] = grouped.apply(
        lambda r: r["market_value"] / r["shares"] if r["shares"] else 0.0, axis=1
    )
    grouped["gain_%"] = grouped.apply(
        lambda r: (r["gain_$"] / r["cost_basis"] * 100) if r["cost_basis"] else 0.0, axis=1
    )
    grouped["day_change_%"] = grouped.apply(
        lambda r: (r["day_change_$"] / (r["market_value"] - r["day_change_$"]) * 100)
                  if (r["market_value"] - r["day_change_$"]) else 0.0,
        axis=1,
    )
    grouped["pct_of_acct"] = grouped["market_value"] / total_mv * 100 if total_mv else 0
    grouped = grouped.sort_values("market_value", ascending=False).reset_index(drop=True)

    # Re-order columns to match what the rest of the UI expects
    keep = ["symbol","name","shares","price","market_value","day_change_$","day_change_%",
            "cost_basis","gain_$","gain_%","pct_of_acct","asset_type"]
    grouped = grouped[[c for c in keep if c in grouped.columns]]

    totals = {
        "market_value": float(grouped["market_value"].sum()),
        "cost_basis":   float(grouped["cost_basis"].sum()),
        "gain_$":       float(grouped["gain_$"].sum()),
        "day_change_$": float(grouped["day_change_$"].sum()),
    }
    totals["gain_%"]       = (totals["gain_$"] / totals["cost_basis"] * 100) if totals["cost_basis"] else 0
    cost_yesterday          = totals["market_value"] - totals["day_change_$"]
    totals["day_change_%"] = (totals["day_change_$"] / cost_yesterday * 100) if cost_yesterday else 0

    return grouped, totals, total_cash


CHATS_DIR    = PORTFOLIO_DIR / "chats"
STRATEGY_DIR = PORTFOLIO_DIR / "strategy"
AI_CACHE_DIR = PORTFOLIO_DIR / "ai_cache"


def _ai_cache_path(key: str) -> Path:
    """Sanitized filesystem path for a cached AI output."""
    safe = _re.sub(r"[^A-Za-z0-9_-]+", "_", key)[:120] or "cache"
    return AI_CACHE_DIR / f"{safe}.txt"


def load_ai_cache(key: str) -> str | None:
    """Read a cached AI output from disk. Returns None if missing/corrupt."""
    path = _ai_cache_path(key)
    if not path.exists():
        return None
    try:
        return path.read_text()
    except Exception:
        return None


def save_ai_cache(key: str, value: str) -> None:
    AI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _ai_cache_path(key).write_text(value)


def invalidate_ai_cache(key: str) -> None:
    """Clear both the in-memory and on-disk copies of a cached AI output."""
    st.session_state.pop(key, None)
    _ai_cache_path(key).unlink(missing_ok=True)


def get_or_cache_ai(key: str, generator) -> str:
    """Standard read-through cache for AI outputs: session_state → disk → generate.
    `generator` is a zero-arg callable that produces the value when both caches miss."""
    if key in st.session_state:
        return st.session_state[key]
    cached = load_ai_cache(key)
    if cached is not None:
        st.session_state[key] = cached
        return cached
    value = generator()
    st.session_state[key] = value
    save_ai_cache(key, value)
    return value


def load_strategy(owner_key: str) -> dict:
    """Load goals/watchlist/theses for one portfolio owner."""
    path = STRATEGY_DIR / f"{owner_key}.json"
    if not path.exists():
        return {"goal": {}, "watchlist": [], "theses": {}}
    try:
        import json
        data = json.loads(path.read_text())
        data.setdefault("goal", {})
        data.setdefault("watchlist", [])
        data.setdefault("theses", {})
        return data
    except Exception:
        return {"goal": {}, "watchlist": [], "theses": {}}


def save_strategy(owner_key: str, data: dict) -> None:
    import json
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    (STRATEGY_DIR / f"{owner_key}.json").write_text(json.dumps(data, indent=2))


def project_goal_progress(current: float, monthly_contrib: float,
                          target: float, years: int, annual_return_pct: float) -> dict:
    """Project portfolio value forward. Returns dict with monthly series + summary."""
    if years <= 0 or target <= 0:
        return {}
    months = years * 12
    r = (annual_return_pct / 100) / 12  # monthly return rate
    series, v = [], current
    for m in range(months + 1):
        series.append({"month": m, "value": v})
        v = v * (1 + r) + monthly_contrib
    final = series[-1]["value"]
    # Required monthly contribution to hit target (PMT formula for FV)
    if r > 0:
        needed = ((target - current * (1 + r) ** months) * r) / ((1 + r) ** months - 1)
    else:
        needed = (target - current) / months
    return {
        "series": series, "final": final, "target": target,
        "on_track": final >= target, "gap": final - target,
        "needed_per_month": needed,
    }


def _chat_path(chat_id: str) -> Path:
    """Stable filesystem path for a chat. chat_id is sanitized."""
    safe = _re.sub(r"[^A-Za-z0-9_-]+", "_", chat_id)[:80] or "chat"
    return CHATS_DIR / f"{safe}.json"


def load_chat(chat_id: str) -> list:
    """Load chat history from disk. Returns [] if file missing/corrupt."""
    path = _chat_path(chat_id)
    if not path.exists():
        return []
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return []


def save_chat(chat_id: str, history: list) -> None:
    """Persist chat history (or delete the file if history is empty)."""
    import json
    path = _chat_path(chat_id)
    if not history:
        path.unlink(missing_ok=True)
        return
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def prune_stock_chats(max_keep: int = 10) -> None:
    """Keep only the N most-recently-modified stock_*.json files."""
    if not CHATS_DIR.exists():
        return
    stock_files = sorted(
        CHATS_DIR.glob("stock_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in stock_files[max_keep:]:
        old.unlink(missing_ok=True)


def load_profile() -> dict:
    """Load investor profile from disk, or return empty dict if missing/corrupt."""
    if not PROFILE_FILE.exists():
        return {}
    try:
        import json
        return json.loads(PROFILE_FILE.read_text())
    except Exception:
        return {}


def save_profile(p: dict) -> None:
    import json
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(p, indent=2))


def load_settings() -> dict:
    """Load UI settings (currently just per-owner display labels)."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        import json
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_settings(s: dict) -> None:
    import json
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def get_owner_label(owner_key: str) -> str:
    """Display label for an owner — user-overridable via settings.json."""
    labels = (load_settings().get("owner_labels") or {})
    return labels.get(owner_key) or DEFAULT_OWNER_LABELS.get(owner_key, owner_key)


def format_profile_for_ai(p: dict) -> str:
    """Render the profile as a block to inject into AI prompts. Returns empty
    string if no profile is set."""
    if not p:
        return ""
    lines = ["=== INVESTOR PROFILE ==="]
    if p.get("age"):              lines.append(f"Age:                  {p['age']}")
    if p.get("horizon_years"):    lines.append(f"Time horizon:         {p['horizon_years']} years until money is needed")
    if p.get("risk"):             lines.append(f"Risk tolerance:       {p['risk']}")
    if p.get("risk_gut"):         lines.append(f"Self-described:       \"{p['risk_gut']}\"")
    if p.get("target_stock") is not None:
        lines.append(
            f"Target allocation:    {p['target_stock']}% stocks / "
            f"{p.get('target_bond', 0)}% bonds / "
            f"{p.get('target_cash', 0)}% cash / "
            f"{p.get('target_other', 0)}% other"
        )
    if p.get("max_sector_pct"):   lines.append(f"Max single sector:    {p['max_sector_pct']}%")
    if p.get("account_type"):     lines.append(f"Account type:         {p['account_type']}")
    if p.get("goal"):             lines.append(f"Primary goal:         {p['goal']}")
    if p.get("monthly_contrib") is not None:
        lines.append(f"Monthly contribution: ${p['monthly_contrib']:,}")
    return "\n".join(lines)

def parse_schwab_csv(file) -> tuple[pd.DataFrame, dict, str | None, float]:
    """Parse a Schwab 'Positions' export. Returns (holdings_df, totals, account_label, cash).
    Strips the account number from the header row — only a generic label like
    'Individual' / 'Roth' is kept."""
    raw = file.read() if hasattr(file, "read") else open(file, encoding="utf-8").read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    lines = [l for l in raw.splitlines() if l.strip()]
    account_label = None
    if lines and lines[0].lstrip('"').lower().startswith("positions for account"):
        m = _re.search(r"account\s+(\S+)", lines[0], _re.IGNORECASE)
        account_label = m.group(1).strip('",') if m else None
        lines = lines[1:]

    df = pd.read_csv(_io.StringIO("\n".join(lines)))
    df = df.dropna(axis=1, how="all").copy()
    df.columns = [c.strip() for c in df.columns]

    totals_row = df[df["Symbol"].str.contains("Positions Total", na=False)]
    cash_row   = df[df["Symbol"].str.contains("Cash", na=False)]
    holdings   = df[~df["Symbol"].isin(
        pd.concat([totals_row["Symbol"], cash_row["Symbol"]])
    )].copy()

    def _num(v):
        if pd.isna(v) or v in ("--", "N/A", ""):
            return None
        s = str(v).replace("$", "").replace(",", "").replace("%", "").strip()
        try: return float(s)
        except Exception: return None

    money = ["Price", "Price Chng $ (Price Change $)", "Mkt Val (Market Value)",
             "Day Chng $ (Day Change $)", "Cost Basis", "Gain $ (Gain/Loss $)"]
    pct   = ["Price Chng % (Price Change %)", "Day Chng % (Day Change %)",
             "Gain % (Gain/Loss %)", "% of Acct (% of Account)"]
    for c in money + pct + ["Qty (Quantity)"]:
        if c in holdings.columns:
            holdings[c] = holdings[c].apply(_num)

    holdings = holdings.rename(columns={
        "Symbol":"symbol","Description":"name","Qty (Quantity)":"shares","Price":"price",
        "Mkt Val (Market Value)":"market_value",
        "Day Chng $ (Day Change $)":"day_change_$","Day Chng % (Day Change %)":"day_change_%",
        "Cost Basis":"cost_basis","Gain $ (Gain/Loss $)":"gain_$",
        "Gain % (Gain/Loss %)":"gain_%","% of Acct (% of Account)":"pct_of_acct",
        "Asset Type":"asset_type",
    })
    keep = ["symbol","name","shares","price","market_value","day_change_$","day_change_%",
            "cost_basis","gain_$","gain_%","pct_of_acct","asset_type"]
    holdings = holdings[[c for c in keep if c in holdings.columns]].reset_index(drop=True)

    totals = {}
    if not totals_row.empty:
        t = totals_row.iloc[0]
        totals = {
            "market_value": _num(t.get("Mkt Val (Market Value)")),
            "cost_basis":   _num(t.get("Cost Basis")),
            "gain_$":       _num(t.get("Gain $ (Gain/Loss $)")),
            "gain_%":       _num(t.get("Gain % (Gain/Loss %)")),
            "day_change_$": _num(t.get("Day Chng $ (Day Change $)")),
            "day_change_%": _num(t.get("Day Chng % (Day Change %)")),
        }

    cash = _num(cash_row.iloc[0].get("Mkt Val (Market Value)")) if not cash_row.empty else 0.0
    return holdings, totals, account_label, (cash or 0.0)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sectors(tickers: tuple[str, ...]) -> dict:
    """Look up Finnhub sector for each ticker. Cached for 1 hour. Returns {ticker: sector}."""
    out = {}
    for t in tickers:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/stock/profile2",
                params={"symbol": t, "token": FINNHUB_API_KEY},
                timeout=8,
            )
            if r.ok:
                out[t] = r.json().get("finnhubIndustry") or "Unknown"
            else:
                out[t] = "Unknown"
        except Exception:
            out[t] = "Unknown"
    return out


def ai_portfolio_review(holdings: pd.DataFrame, totals: dict, cash: float,
                        sectors: dict, profile: dict | None = None,
                        strategy: dict | None = None) -> str:
    """Concentration risk, sector tilt, performance, what to watch.
    If profile is provided, advice is tailored to it; otherwise Claude opens
    with a warning that it's giving generic advice.
    If strategy is provided, advice also references the user's goal/watchlist/theses."""
    if holdings.empty:
        return "No holdings to analyze."

    profile = profile or {}
    strategy = strategy or {}
    profile_block = format_profile_for_ai(profile)
    profile_caveat = (
        ""
        if profile_block
        else "IMPORTANT: No investor profile is set (no age, horizon, risk tolerance, "
             "or target allocation). Begin your response with one short paragraph noting "
             "this means your advice is generic and not tailored to the user's situation, "
             "then continue with the analysis anyway.\n\n"
    )

    total_mv = totals.get("market_value") or holdings["market_value"].sum()
    total_cb = totals.get("cost_basis")   or holdings["cost_basis"].sum()
    gain_pct = totals.get("gain_%")       or ((total_mv / total_cb - 1) * 100 if total_cb else 0)
    day_pct  = totals.get("day_change_%") or 0

    # Per-holding lines for the prompt, with user's stated thesis (if any)
    theses = strategy.get("theses") or {}
    rows = []
    for _, r in holdings.iterrows():
        sym = r["symbol"]
        sector = sectors.get(sym, "Unknown")
        line = (
            f"- {sym} ({r['name']}, {sector}): "
            f"${r['market_value']:.2f} = {r['pct_of_acct']:.1f}% of portfolio, "
            f"day {r['day_change_%']:+.2f}%, total return {r['gain_%']:+.1f}%"
        )
        if theses.get(sym):
            line += f"  | THESIS: {theses[sym]}"
        rows.append(line)
    holdings_block = "\n".join(rows)

    # Sector breakdown
    by_sector = holdings.assign(sector=holdings["symbol"].map(sectors)) \
        .groupby("sector")["market_value"].sum().sort_values(ascending=False)
    sector_block = "\n".join(f"- {s}: ${v:.2f} ({v/total_mv*100:.1f}%)" for s, v in by_sector.items())

    # Goal + watchlist context for the prompt
    goal = strategy.get("goal") or {}
    watchlist = strategy.get("watchlist") or []
    goal_block = ""
    if goal.get("target"):
        proj = project_goal_progress(
            current=float(total_mv),
            monthly_contrib=float(profile.get("monthly_contrib") or 0),
            target=float(goal["target"]),
            years=int(goal.get("years") or 20),
            annual_return_pct=float(goal.get("annual_return_pct") or 7.0),
        )
        if proj:
            status = "ON TRACK" if proj["on_track"] else "OFF TRACK"
            goal_block = (
                f"\nGOAL:\n"
                f"- Target: ${goal['target']:,} in {goal.get('years', 20)} years "
                f"@ assumed {goal.get('annual_return_pct', 7.0)}% return\n"
                f"- Monthly contribution: ${float(profile.get('monthly_contrib') or 0):,.0f}\n"
                f"- Projected final value: ${proj['final']:,.0f}  ({status}, "
                f"{'surplus' if proj['on_track'] else 'gap'} ${abs(proj['gap']):,.0f})\n"
                f"- Required monthly contribution to hit target: ${proj['needed_per_month']:,.0f}\n"
            )
    watch_block = ""
    if watchlist:
        wlines = [
            f"- {w.get('symbol', '?')}" + (f": {w['note']}" if w.get("note") else "")
            for w in watchlist
        ]
        watch_block = "\nWATCHLIST (not owned, user is considering):\n" + "\n".join(wlines) + "\n"

    has_goal      = bool(goal_block)
    has_watchlist = bool(watch_block)
    has_theses    = bool(theses)

    extra_sections = ""
    if has_goal:
        extra_sections += (
            "\n### Goal check\n"
            "1-2 sentences on whether the user is on track for their stated goal "
            "(see GOAL block). If off track, name the size of the gap and what would "
            "close it (higher contribution, more time, different return assumption). "
            "If their assumed return looks aggressive given their risk tolerance, say so.\n"
        )
    if has_watchlist:
        extra_sections += (
            "\n### Watchlist take\n"
            "Briefly comment on the watchlist names (see WATCHLIST block). Would adding "
            "any of them help with concentration/sector gaps you flagged above? Be honest "
            "if a name doesn't fit the portfolio. Don't endorse a name just because it's listed.\n"
        )
    theses_instruction = (
        " Each holding line includes the user's THESIS where they wrote one — "
        "weigh whether the thesis still seems to hold given the numbers, and flag holdings "
        "with no thesis as ones the user should articulate."
        if has_theses else ""
    )

    prompt = f"""{profile_caveat}You are reviewing a real personal investment portfolio.
Be specific and honest — flag actual risks, don't be generic. {"Tailor your analysis to the investor's profile below (age, horizon, risk tolerance, target allocation, goal, account type)." if profile_block else "Note: no investor profile, so keep advice general."}{theses_instruction}

Write in sections with these exact headers (do NOT add a title above them):

### Portfolio at a glance
2-3 sentences: total value, total return, today's move. {"Compare actual allocation to their stated target." if profile.get('target_stock') is not None else ""} Note anything striking.

### Concentration & sector risk
This is the most important section. Look at how the portfolio is allocated.
Call out specific risks: single-stock concentration above 25%, single-sector
concentration above {profile.get('max_sector_pct', 60)}%, lack of diversification, etc. Quote the actual numbers.
{"Reference their max-sector cap explicitly." if profile.get('max_sector_pct') else ""}
{extra_sections}
### What to watch / suggestions
2-3 concrete things to consider. Phrase as questions or considerations, not
commands. {"Tailor to their time horizon and account type — e.g. for taxable accounts mention tax implications; for long horizons emphasize that volatility matters less." if profile_block else ""}
End with: "Not financial advice."

{profile_block}

PORTFOLIO DATA:
- Total market value: ${total_mv:,.2f}
- Total cost basis:   ${total_cb:,.2f}
- Total gain:         {gain_pct:+.1f}%
- Today:              {day_pct:+.2f}%
- Cash:               ${cash:.2f}
- # of holdings:      {len(holdings)}
{goal_block}{watch_block}
HOLDINGS:
{holdings_block}

BY SECTOR:
{sector_block}
"""
    return call_claude(prompt, max_tokens=1200)


def build_portfolio_chat_context(holdings: pd.DataFrame, totals: dict, cash: float,
                                 sectors: dict, review: str | None,
                                 profile: dict | None = None,
                                 strategy: dict | None = None) -> str:
    """System prompt for the portfolio chatbot — same idea as the stock one,
    but holds the user's actual holdings instead of a single ticker.
    If profile is provided, advice can reference time horizon, risk, targets, etc.
    If strategy is provided, advice can reference goals/watchlist/theses too."""
    profile = profile or {}
    strategy = strategy or {}
    profile_block = format_profile_for_ai(profile)
    total_mv = totals.get("market_value") or holdings["market_value"].sum()
    total_cb = totals.get("cost_basis")   or holdings["cost_basis"].sum()
    gain_pct = totals.get("gain_%")       or ((total_mv / total_cb - 1) * 100 if total_cb else 0)

    theses = strategy.get("theses") or {}
    rows = []
    for _, r in holdings.iterrows():
        sec = sectors.get(r["symbol"], "Unknown")
        line = (
            f"- {r['symbol']} ({r['name']}, {sec}): "
            f"${r['market_value']:.2f} = {r['pct_of_acct']:.1f}% of portfolio, "
            f"day {r['day_change_%']:+.2f}%, total return {r['gain_%']:+.1f}%"
        )
        if theses.get(r["symbol"]):
            line += f"  | THESIS: {theses[r['symbol']]}"
        rows.append(line)
    holdings_block = "\n".join(rows)

    by_sector = holdings.assign(sector=holdings["symbol"].map(sectors)) \
        .groupby("sector")["market_value"].sum().sort_values(ascending=False)
    sector_block = "\n".join(f"- {s}: ${v:.2f} ({v/total_mv*100:.1f}%)" for s, v in by_sector.items())

    parts = [
        "You are a financial assistant embedded in a personal portfolio dashboard.",
        "Answer the user's questions about THEIR portfolio (data below). Use real numbers, not generic advice.",
        "Be concise (2-4 short paragraphs). Define jargon when you use it.",
        "If asked about a stock not in their portfolio, say so but answer briefly with general info if you can.",
        "End with 'Not financial advice.' if you make any directional or trade-related call.",
    ]
    if profile_block:
        parts += [
            "",
            "Use the investor profile below to tailor your answers — reference their time horizon,",
            "risk tolerance, target allocation, and account type explicitly when relevant.",
            "",
            profile_block,
        ]
    else:
        parts += [
            "",
            "NOTE: No investor profile is set. When the user asks for portfolio advice, briefly note",
            "you don't know their age/horizon/risk profile, then give a general answer. Don't refuse.",
        ]
    parts += [
        "",
        "=== PORTFOLIO TOTALS ===",
        f"Total value: ${total_mv:,.2f}",
        f"Cost basis:  ${total_cb:,.2f}",
        f"Total gain:  {gain_pct:+.1f}%",
        f"Today:       {totals.get('day_change_%', 0):+.2f}%",
        f"Cash:        ${cash:,.2f}",
        f"# holdings:  {len(holdings)}",
        "",
        "=== HOLDINGS (with stated theses where the user wrote one) ===",
        holdings_block,
        "",
        "=== BY SECTOR ===",
        sector_block,
    ]

    goal = strategy.get("goal") or {}
    if goal.get("target"):
        proj = project_goal_progress(
            current=float(total_mv),
            monthly_contrib=float(profile.get("monthly_contrib") or 0),
            target=float(goal["target"]),
            years=int(goal.get("years") or 20),
            annual_return_pct=float(goal.get("annual_return_pct") or 7.0),
        )
        if proj:
            status = "ON TRACK" if proj["on_track"] else "OFF TRACK"
            parts += [
                "",
                "=== GOAL ===",
                f"Target: ${goal['target']:,} in {goal.get('years', 20)} years "
                f"@ assumed {goal.get('annual_return_pct', 7.0)}% return",
                f"Monthly contribution: ${float(profile.get('monthly_contrib') or 0):,.0f}",
                f"Projected: ${proj['final']:,.0f} ({status}, "
                f"{'surplus' if proj['on_track'] else 'gap'} ${abs(proj['gap']):,.0f})",
                f"Required monthly to hit target: ${proj['needed_per_month']:,.0f}",
            ]

    watchlist = strategy.get("watchlist") or []
    if watchlist:
        wlines = [
            f"- {w.get('symbol', '?')}" + (f": {w['note']}" if w.get("note") else "")
            for w in watchlist
        ]
        parts += ["", "=== WATCHLIST (not owned, user is considering) ===", *wlines]

    if review:
        parts += ["", "=== AI REVIEW ALREADY SHOWN TO USER ===", review]
    return "\n".join(parts)


# ============================================================
# Market Today: aggregated news + earnings + AI briefing
# ============================================================

import feedparser as _feedparser

# Free RSS feeds. Reuters discontinued public RSS in 2020 — left out.
# WSJ headlines + summaries are public; full articles are paywalled, which is fine.
RSS_FEEDS = {
    "CNBC top news":  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC markets":   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "MarketWatch":    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance":  "https://finance.yahoo.com/news/rssindex",
    "WSJ markets":    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_news() -> tuple[list, dict]:
    """Aggregate market news from Finnhub general + all RSS feeds. Each item is
    a dict {source, title, url, summary, ts}. Returns (items, errors_by_source)."""
    items, errors = [], {}

    # Finnhub general news
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        for n in r.json()[:30]:
            items.append({
                "source":  n.get("source") or "Finnhub",
                "title":   n.get("headline", ""),
                "url":     n.get("url", ""),
                "summary": n.get("summary", "")[:300],
                "ts":      n.get("datetime") or 0,
            })
    except Exception as e:
        errors["Finnhub"] = f"{type(e).__name__}: {e}"

    # RSS feeds
    for name, url in RSS_FEEDS.items():
        try:
            f = _feedparser.parse(url)
            if not f.entries:
                errors[name] = "empty feed"
                continue
            for e in f.entries[:15]:
                # Try several common timestamp fields
                ts = 0
                for key in ("published_parsed", "updated_parsed"):
                    if e.get(key):
                        import time as _t
                        ts = int(_t.mktime(e[key]))
                        break
                summary = (e.get("summary") or "").strip()
                # Strip HTML tags from RSS summaries (some feeds embed them)
                summary = _re.sub(r"<[^>]+>", "", summary)[:300]
                items.append({
                    "source":  name,
                    "title":   e.get("title", "").strip(),
                    "url":     e.get("link", ""),
                    "summary": summary,
                    "ts":      ts,
                })
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"

    # Dedupe by title (different sources sometimes carry the same story)
    seen, dedup = set(), []
    for item in items:
        key = (item["title"] or "").lower().strip()[:80]
        if key and key not in seen:
            seen.add(key)
            dedup.append(item)

    # Sort by timestamp desc (items with no ts go to the bottom)
    dedup.sort(key=lambda x: x["ts"] or 0, reverse=True)
    return dedup, errors


@st.cache_data(ttl=900, show_spinner=False)
def fetch_earnings_week(portfolio_tickers: tuple[str, ...] = ()) -> list:
    """Earnings in the next 7 days. Filters down to a manageable set: large/
    well-known names + any tickers in the user's portfolio."""
    from datetime import date, timedelta
    fr = date.today().isoformat()
    to = (date.today() + timedelta(days=7)).isoformat()
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": fr, "to": to, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        all_earnings = r.json().get("earningsCalendar", [])
    except Exception:
        return []

    # Hand-picked watchlist of widely-followed names. We don't have market cap
    # data here, so use a heuristic list of the most-talked-about tickers.
    WATCHLIST = {
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        "AVGO", "ORCL", "AMD", "NFLX", "CRM", "ADBE", "INTC", "QCOM",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
        "BRK.B", "V", "MA", "PYPL", "WMT", "COST", "HD", "TGT",
        "JNJ", "PFE", "MRK", "LLY", "ABBV", "UNH", "TMO", "ABT",
        "XOM", "CVX", "COP", "BA", "CAT", "GE", "F", "GM",
        "DIS", "NKE", "SBUX", "MCD", "KO", "PEP", "PG",
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    }
    keep = WATCHLIST | set(portfolio_tickers)

    filtered = [e for e in all_earnings if e.get("symbol") in keep]
    # Sort by date, then symbol
    filtered.sort(key=lambda e: (e.get("date", ""), e.get("symbol", "")))
    return filtered


def ai_market_briefing(news: list, earnings: list, portfolio_tickers: tuple = ()) -> str:
    """3-paragraph daily market briefing. References user's portfolio if provided."""
    if not news:
        return "No market news available right now."

    headlines = "\n".join(
        f"- [{n['source']}] {n['title']}" for n in news[:25] if n.get("title")
    )
    earnings_block = "(no major earnings in the next 7 days)"
    if earnings:
        earnings_block = "\n".join(
            f"- {e['symbol']} on {e['date']} ({e.get('hour','')})"
            for e in earnings[:15]
        )
    portfolio_line = (
        f"User's portfolio holdings (cross-reference if relevant): {', '.join(portfolio_tickers)}"
        if portfolio_tickers else "User has not uploaded a portfolio yet."
    )

    prompt = f"""You are a market analyst writing the daily briefing for someone who
will read this once and decide what to pay attention to. Write 3 short paragraphs
(80-120 words each) with these exact headers:

### Today's dominant narrative
What's the single story driving markets today? Be specific — name the catalyst,
the sectors moving, the index direction if mentioned in the headlines. Don't be
vague.

### Sectors and names in motion
What's moving and why? Group by theme (e.g. "AI infrastructure," "energy after
OPEC headlines"). If the user has portfolio holdings (see below), explicitly
call out any of THEIR names that appear in the news.

### What to watch
2-3 specific things to track — upcoming earnings, data releases, geopolitical
catalysts mentioned in the news. Keep it actionable.

Be honest. If the news is mixed or directionless, say so. End with: "Not financial advice."

{portfolio_line}

HEADLINES (most recent first, source in brackets):
{headlines}

EARNINGS THIS WEEK (large-cap + user's holdings):
{earnings_block}
"""
    return call_claude(prompt, max_tokens=1000)


def build_market_chat_context(news: list, earnings: list, briefing: str | None,
                              portfolio_tickers: tuple = (),
                              profile: dict | None = None) -> str:
    """System prompt for the market chat."""
    headlines = "\n".join(
        f"- [{n['source']}] {n['title']}" for n in news[:30] if n.get("title")
    )
    earnings_block = "\n".join(
        f"- {e['symbol']} on {e['date']}"
        for e in (earnings or [])[:15]
    ) or "(none in next 7 days from watchlist)"

    parts = [
        "You are a market news assistant. Answer questions using ONLY the news headlines below.",
        "If a question requires details from a full article you don't have, say so.",
        "Be concise (2-4 short paragraphs). Define jargon when you use it.",
        f"User's portfolio: {', '.join(portfolio_tickers) if portfolio_tickers else '(not uploaded)'}",
        "When relevant, cross-reference news against their portfolio holdings.",
        "End with 'Not financial advice.' on any directional call.",
    ]
    if profile:
        parts += ["", format_profile_for_ai(profile)]
    parts += [
        "",
        "=== TODAY'S HEADLINES (most recent first) ===",
        headlines,
        "",
        "=== EARNINGS THIS WEEK ===",
        earnings_block,
    ]
    if briefing:
        parts += ["", "=== AI BRIEFING ALREADY SHOWN TO USER ===", briefing]
    return "\n".join(parts)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📈 Briefing Markets")

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

st.sidebar.markdown("**Quick picks**")
cols = st.sidebar.columns(4)
for i, t in enumerate(PRESET_TICKERS):
    if cols[i % 4].button(t, use_container_width=True, key=f"preset_{t}"):
        st.session_state.ticker = t

st.sidebar.markdown("**Or type any ticker**")
typed = st.sidebar.text_input("Ticker", value=st.session_state.ticker, label_visibility="collapsed")
if typed and typed.upper() != st.session_state.ticker:
    st.session_state.ticker = typed.upper()

period = st.sidebar.selectbox(
    "Chart range",
    list(PERIOD_DAYS.keys()),
    index=3,
)

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "**Sources**  \n"
    "• Quote/news/fundamentals: Finnhub  \n"
    "• Chart bars: Alpaca IEX feed  \n"
    "• AI: Claude Haiku 4.5 via OpenRouter"
)
st.sidebar.caption(
    "**Theme:** flip light/dark via the top-right `⋮` menu → Settings → Theme."
)

ticker = st.session_state.ticker


# ============================================================
# Tabs: single-stock view + portfolio view
# ============================================================

mine_label = get_owner_label("mine")
moms_label = get_owner_label("moms")
tab_stock, tab_mine, tab_moms, tab_market = st.tabs(
    ["📊 Single Stock", f"💼 {mine_label}", f"👥 {moms_label}", "🌍 Market Today"]
)

with tab_stock:
    # ============================================================
    # Fetch
    # ============================================================

    with st.spinner(f"Loading {ticker}…"):
        quote, profile, metrics, news, fh_errors = fetch_finnhub(ticker)
        bars = fetch_bars(ticker, PERIOD_DAYS[period])

    # Profile is the canonical "is this a real ticker" check
    if not profile and quote.get("c", 0) == 0:
        st.error(f"Couldn't load data for **{ticker}**. Check the ticker symbol.")
        if fh_errors:
            with st.expander("Error details"):
                for src, msg in fh_errors.items():
                    st.code(f"{src}: {msg}")
        if st.button("Clear cache and retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    ind = compute_indicators(bars)


    # ============================================================
    # Header + top metrics
    # ============================================================

    name = profile.get("name") or ticker
    st.title(f"{ticker} — {name}")
    st.caption(
        f"{profile.get('finnhubIndustry', '')} · {profile.get('exchange', '')} · "
        f"IPO {profile.get('ipo', 'n/a')}"
    )

    price       = quote.get("c")
    day_change  = quote.get("d")
    day_pct     = quote.get("dp")
    mkt_cap_m   = profile.get("marketCapitalization")  # in millions
    mkt_cap     = mkt_cap_m * 1e6 if mkt_cap_m else None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Price",
        fmt_money(price),
        f"{day_change:+.2f} ({day_pct:+.2f}%)" if day_change is not None else None,
    )
    m2.metric("Market cap", fmt_big(mkt_cap))
    m3.metric("P/E (TTM)", f"{metrics.get('peTTM'):.1f}" if metrics.get("peTTM") else "—")
    m4.metric("52w high", fmt_money(metrics.get("52WeekHigh")))
    m5.metric("52w low", fmt_money(metrics.get("52WeekLow")))

    st.divider()


    # ============================================================
    # Chart — price + MAs + Bollinger, RSI, MACD
    # ============================================================

    st.subheader(f"Price chart with technical indicators ({period})")

    if bars.empty:
        st.info("No price history available from Alpaca for this ticker.")
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.04,
            subplot_titles=("Price + Moving Averages + Bollinger Bands", "RSI(14)", "MACD"),
        )

        # --- Row 1: price + MAs + Bollinger ---
        fig.add_trace(go.Scatter(
            x=bars.index, y=bars["close"], name="Close",
            line=dict(width=2, color="#6366f1"),
        ), row=1, col=1)

        if ind:
            s = ind["series"]
            fig.add_trace(go.Scatter(x=bars.index, y=s["ma20"],  name="MA20",
                                     line=dict(width=1, color="#10b981")), row=1, col=1)
            fig.add_trace(go.Scatter(x=bars.index, y=s["ma50"],  name="MA50",
                                     line=dict(width=1, color="#f59e0b")), row=1, col=1)
            fig.add_trace(go.Scatter(x=bars.index, y=s["ma200"], name="MA200",
                                     line=dict(width=1, color="#ef4444")), row=1, col=1)
            # Bollinger as a shaded band
            fig.add_trace(go.Scatter(x=bars.index, y=s["bb_up"], name="BB upper",
                                     line=dict(width=0.5, color="rgba(150,150,150,0.5)"),
                                     showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=bars.index, y=s["bb_dn"], name="BB lower",
                                     line=dict(width=0.5, color="rgba(150,150,150,0.5)"),
                                     fill="tonexty", fillcolor="rgba(150,150,150,0.08)",
                                     showlegend=False), row=1, col=1)

            # --- Row 2: RSI ---
            fig.add_trace(go.Scatter(x=bars.index, y=s["rsi"], name="RSI",
                                     line=dict(width=1.5, color="#8b5cf6"),
                                     showlegend=False), row=2, col=1)
            fig.add_hline(y=70, line=dict(dash="dash", width=1, color="#ef4444"), row=2, col=1)
            fig.add_hline(y=30, line=dict(dash="dash", width=1, color="#10b981"), row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)

            # --- Row 3: MACD ---
            fig.add_trace(go.Scatter(x=bars.index, y=s["macd"], name="MACD",
                                     line=dict(width=1.5, color="#3b82f6"),
                                     showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=bars.index, y=s["signal"], name="Signal",
                                     line=dict(width=1, color="#f59e0b"),
                                     showlegend=False), row=3, col=1)
            colors = ["#10b981" if v >= 0 else "#ef4444" for v in s["hist"].fillna(0)]
            fig.add_trace(go.Bar(x=bars.index, y=s["hist"], name="Histogram",
                                 marker_color=colors, opacity=0.5,
                                 showlegend=False), row=3, col=1)

        fig.update_layout(
            height=760,
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.06,
                xanchor="left", x=0,
            ),
        )
        fig.update_yaxes(tickprefix="$", row=1, col=1)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},  # hide the toolbar that overlaps the legend
        )

    st.divider()


    # ============================================================
    # Two-column: stats + AI sections
    # ============================================================

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Key stats")
        stats = {
            "Open":          fmt_money(quote.get("o")),
            "Day high":      fmt_money(quote.get("h")),
            "Day low":       fmt_money(quote.get("l")),
            "Prev close":    fmt_money(quote.get("pc")),
            "EPS (TTM)":     fmt_money(metrics.get("epsTTM")),
            "Beta":          f"{metrics.get('beta'):.2f}" if metrics.get("beta") else "—",
            "Dividend yld":  f"{metrics.get('dividendYieldIndicatedAnnual'):.2f}%" if metrics.get("dividendYieldIndicatedAnnual") else "—",
            "Shares out":    fmt_big(profile.get("shareOutstanding") * 1e6) if profile.get("shareOutstanding") else "—",
        }
        st.table(pd.DataFrame(stats.items(), columns=["", ""]).set_index(""))

        if ind:
            st.subheader("Indicator snapshot")
            L = ind["latest"]
            snap = {
                "RSI(14)":       f"{L['rsi']:.1f}" if L["rsi"] else "—",
                "MACD":          f"{L['macd']:.3f}" if L["macd"] else "—",
                "MACD signal":   f"{L['signal']:.3f}" if L["signal"] else "—",
                "vs 20-day MA":  "above" if L["above_20ma"] else "below",
                "vs 50-day MA":  "above" if L["above_50ma"] else "below",
                "vs 200-day MA": "above" if L["above_200ma"] else "below",
                "Vol vs avg":    f"{L['vol_ratio']:.2f}x" if L["vol_ratio"] else "—",
            }
            st.table(pd.DataFrame(snap.items(), columns=["", ""]).set_index(""))

    with right:
        # --- AI news briefing ---
        st.subheader("🤖 News briefing")
        bcol, btn = st.columns([5, 1])
        with btn:
            refresh_b = st.button("Refresh", key="refresh_brief", use_container_width=True)

        cache_b = f"brief_{ticker}"
        if refresh_b:
            invalidate_ai_cache(cache_b)
        with st.spinner("Reading the news…"):
            briefing_text = get_or_cache_ai(
                cache_b,
                lambda: ai_news_briefing(ticker, profile, quote, news),
            )
        with bcol:
            st.caption("✨ AI generated")
            st.markdown(briefing_text.replace("$", "\\$"))

        st.markdown("")  # spacer

        # --- AI technical analysis ---
        st.subheader("📊 Technical analysis (beginner mode)")
        tcol, tbtn = st.columns([5, 1])
        with tbtn:
            refresh_t = st.button("Refresh", key="refresh_ta", use_container_width=True)

        cache_t = f"ta_{ticker}_{period}"
        if refresh_t:
            invalidate_ai_cache(cache_t)
        with st.spinner("Reading the chart…"):
            ta_text = get_or_cache_ai(
                cache_t,
                lambda: ai_technical_analysis(ticker, profile, ind),
            )
        with tcol:
            st.caption("✨ AI generated")
            st.markdown(ta_text.replace("$", "\\$"))

    st.divider()


    # ============================================================
    # News
    # ============================================================

    st.subheader(f"Recent news ({len(news)} items)")
    if not news:
        st.caption("No recent news found.")
    else:
        for item in news[:15]:
            title     = item.get("headline") or "(untitled)"
            source    = item.get("source", "")
            url       = item.get("url", "")
            ts        = item.get("datetime")
            when      = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
            summary   = item.get("summary", "")[:200]

            st.markdown(f"**[{title}]({url})**  \n_{source} · {when}_")
            if summary:
                st.caption(summary + ("…" if len(item.get("summary", "")) > 200 else ""))
            st.markdown("")


def render_portfolio_tab(owner_key: str, owner_label: str):
    """Render a portfolio tab for one owner. owner_key is a stable identifier
    used for storage paths + session_state keys (e.g. 'mine', 'moms').
    owner_label is the display name (e.g. 'My Portfolio')."""

    csv_path        = PORTFOLIOS_DIR / f"{owner_key}.csv"
    sess_data_key   = f"portfolio_{owner_key}"
    sess_review_key = f"portfolio_review_{owner_key}"
    sess_chat_key   = f"portfolio_chat_{owner_key}"

    st.subheader(f"💼 {owner_label}")
    st.caption(
        "Upload a Schwab Positions CSV. Account number is stripped on parse — "
        "only tickers, shares, and dollar amounts are kept. Auto-saved locally."
    )

    # The investor profile expander lives in 'My Portfolio' only.
    # Mom's portfolio uses the same profile (you're managing both).
    if owner_key == "mine":
        if "profile" not in st.session_state:
            st.session_state.profile = load_profile()
        profile = st.session_state.profile
        profile_set = bool(profile)
        profile_label = "✅ Your profile" if profile_set else "⚠️ Set up your profile (untailored advice without it)"

        with st.expander(profile_label, expanded=not profile_set):
            st.caption(
                "Tells the AI your goals, horizon, risk tolerance, and target allocation. "
                "Without this, advice is generic. Applies to both portfolio tabs. Saved to "
                f"`{PROFILE_FILE}`."
            )
            with st.form(f"profile_form_{owner_key}"):
                f1, f2 = st.columns(2)
                age = f1.number_input("Age", min_value=0, max_value=120,
                                      value=int(profile.get("age", 30)))
                horizon = f2.number_input("Years until you need this money",
                                          min_value=0, max_value=80,
                                          value=int(profile.get("horizon_years", 20)),
                                          help="Retirement at 65 and you're 30? That's 35.")

                f3, f4 = st.columns(2)
                risk = f3.selectbox(
                    "Risk tolerance",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(profile.get("risk", "Medium")),
                )
                risk_gut = f4.text_input(
                    "Honest gut check",
                    value=profile.get("risk_gut", ""),
                    placeholder="e.g. 'I'd hold through a 30% drawdown but panic at 50%'",
                )

                st.markdown("**Target allocation** (should sum to 100)")
                t1, t2, t3, t4 = st.columns(4)
                target_stock = t1.number_input("% Stocks", 0, 100,
                                               value=int(profile.get("target_stock", 90)))
                target_bond  = t2.number_input("% Bonds", 0, 100,
                                               value=int(profile.get("target_bond", 5)))
                target_cash  = t3.number_input("% Cash", 0, 100,
                                               value=int(profile.get("target_cash", 5)))
                target_other = t4.number_input("% Other", 0, 100,
                                               value=int(profile.get("target_other", 0)))

                f5, f6 = st.columns(2)
                max_sector = f5.number_input(
                    "Max % in any single sector",
                    min_value=10, max_value=100,
                    value=int(profile.get("max_sector_pct", 40)),
                    help="Common rule of thumb: cap at 25-40% per sector.",
                )
                account_type = f6.selectbox(
                    "Account type",
                    ["Taxable brokerage", "Roth IRA", "Traditional IRA",
                     "401(k)", "HSA", "Other"],
                    index=["Taxable brokerage", "Roth IRA", "Traditional IRA",
                           "401(k)", "HSA", "Other"].index(
                        profile.get("account_type", "Taxable brokerage")
                    ),
                )

                f7, f8 = st.columns(2)
                goal = f7.selectbox(
                    "Primary goal",
                    ["Long-term wealth building", "Retirement",
                     "Major purchase (house/car)", "Income generation",
                     "Capital preservation", "Other"],
                    index=["Long-term wealth building", "Retirement",
                           "Major purchase (house/car)", "Income generation",
                           "Capital preservation", "Other"].index(
                        profile.get("goal", "Long-term wealth building")
                    ),
                )
                monthly_contrib = f8.number_input(
                    "Monthly contribution ($)",
                    min_value=0, max_value=100000, step=50,
                    value=int(profile.get("monthly_contrib", 0)),
                )

                saved = st.form_submit_button("Save profile", use_container_width=True)

            if saved:
                total = target_stock + target_bond + target_cash + target_other
                if total != 100:
                    st.error(f"Target allocation must sum to 100 — currently {total}.")
                else:
                    new_profile = {
                        "age": age, "horizon_years": horizon,
                        "risk": risk, "risk_gut": risk_gut.strip(),
                        "target_stock": target_stock, "target_bond": target_bond,
                        "target_cash": target_cash, "target_other": target_other,
                        "max_sector_pct": max_sector,
                        "account_type": account_type, "goal": goal,
                        "monthly_contrib": monthly_contrib,
                    }
                    save_profile(new_profile)
                    st.session_state.profile = new_profile
                    # Invalidate cached AI outputs that depend on the profile:
                    # portfolio reviews/chats (both in-memory and on-disk) and
                    # any stock-tab briefings/TA in session_state.
                    for k in list(st.session_state.keys()):
                        if k.startswith(("portfolio_review_", "portfolio_chat_",
                                         "brief_", "ta_")):
                            st.session_state.pop(k, None)
                    if AI_CACHE_DIR.exists():
                        for f in AI_CACHE_DIR.glob("portfolio_review_*.txt"):
                            f.unlink(missing_ok=True)
                        for f in AI_CACHE_DIR.glob("brief_*.txt"):
                            f.unlink(missing_ok=True)
                        for f in AI_CACHE_DIR.glob("ta_*.txt"):
                            f.unlink(missing_ok=True)
                    st.success("Profile saved. AI outputs will refresh.")
                    st.rerun()

    # --- Auto-load saved CSV on first session render ---
    if sess_data_key not in st.session_state and csv_path.exists():
        try:
            raw = csv_path.read_bytes()
            h, t, _label, c = parse_schwab_csv(_io.BytesIO(raw))
            mtime = datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.session_state[sess_data_key] = {
                "holdings": h, "totals": t, "cash": c,
                "source": f"loaded from {csv_path.name} (last updated {mtime})",
            }
        except Exception as e:
            st.warning(f"Couldn't load saved file {csv_path.name}: {e}")

    # --- File uploads ---
    # Each portfolio supports two CSV types from Schwab. Only Positions is required.
    has_data = sess_data_key in st.session_state

    st.markdown("### 📁 Data files")
    st.caption(
        "**Positions** = what you own *right now* (required). "
        "**Transaction History** = every buy/sell with dates (optional — unlocks tax-aware advice)."
    )

    # ---- Positions CSV (required) ----
    label1 = "1️⃣ Positions CSV — ✅ loaded" if has_data else "1️⃣ Positions CSV — required"
    with st.expander(label1, expanded=not has_data):
        st.caption(
            "**What it is:** Snapshot of your current holdings — symbol, shares, price, cost basis, gains.  \n"
            "**Where to get it:** Schwab → Accounts → Positions → top-right Export icon → CSV.  \n"
            "**What we do with it:** Build the holdings table, allocation charts, and AI review."
        )
        upload = st.file_uploader(
            "Positions CSV",
            type=["csv"],
            key=f"upload_positions_{owner_key}",
            help="Overwrites any previously uploaded Positions CSV for this portfolio.",
        )
        if upload is not None:
            try:
                raw_bytes = upload.read()
                PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
                csv_path.write_bytes(raw_bytes)
                h, t, _label, c = parse_schwab_csv(_io.BytesIO(raw_bytes))
                st.session_state[sess_data_key] = {
                    "holdings": h, "totals": t, "cash": c,
                    "source": f"uploaded · saved to {csv_path.name}",
                }
                invalidate_ai_cache(sess_review_key)
                st.session_state.pop(sess_chat_key, None)
                st.success(f"Positions saved as {csv_path.name}.")
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't parse that CSV: {e}")

    # ---- Transaction History CSV (optional, not yet wired up) ----
    with st.expander("2️⃣ Transaction History CSV — optional · coming soon", expanded=False):
        st.caption(
            "**What it is:** Every buy / sell / dividend / reinvestment, with dates.  \n"
            "**Where to get it:** Schwab → History & Statements → Transactions → Export → CSV.  \n"
            "**What we'll do with it:** Per-lot purchase dates, short-vs-long-term gains breakdown, tax-aware advice.  \n"
            "**Status:** Parser not built yet. Share a sample CSV with Claude and it'll wire this up next."
        )
        st.info(
            "🚧 Not active yet. The AI review currently uses Positions only — Transaction History "
            "support is the next thing on the build list."
        )

    if not has_data:
        st.info("👆 Upload your **Positions CSV** above to see this portfolio.")
        return

    # --- Render the portfolio ---
    p = st.session_state[sess_data_key]
    holdings, p_totals, cash = p["holdings"], p["totals"], p["cash"]

    bits = [f"📂 {owner_label}", f"{len(holdings)} holdings", f"${cash:,.2f} cash"]
    if p.get("source"):
        bits.append(p["source"])
    st.caption(" · ".join(bits))

    # Top metrics
    total_mv = p_totals.get("market_value") or holdings["market_value"].sum()
    total_cb = p_totals.get("cost_basis")   or holdings["cost_basis"].sum()
    total_g  = p_totals.get("gain_$")       or (total_mv - total_cb)
    total_gp = p_totals.get("gain_%")       or ((total_mv / total_cb - 1) * 100 if total_cb else 0)
    day_d    = p_totals.get("day_change_$") or 0
    day_p    = p_totals.get("day_change_%") or 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total value", f"${total_mv:,.2f}", f"{day_d:+,.2f} ({day_p:+.2f}%) today")
    m2.metric("Cost basis", f"${total_cb:,.2f}")
    m3.metric("Total gain", f"${total_g:+,.2f}", f"{total_gp:+.2f}%")
    m4.metric("Holdings", f"{len(holdings)}")

    st.divider()

    # Holdings table
    st.subheader("Holdings")
    display = holdings.copy()
    display["market_value"]  = display["market_value"].apply(lambda v: f"${v:,.2f}")
    display["cost_basis"]    = display["cost_basis"].apply(lambda v: f"${v:,.2f}" if v else "—")
    display["price"]         = display["price"].apply(lambda v: f"${v:,.2f}")
    display["gain_$"]        = display["gain_$"].apply(lambda v: f"${v:+,.2f}" if v is not None else "—")
    display["gain_%"]        = display["gain_%"].apply(lambda v: f"{v:+.2f}%" if v is not None else "—")
    display["day_change_$"]  = display["day_change_$"].apply(lambda v: f"${v:+,.2f}" if v is not None else "—")
    display["day_change_%"]  = display["day_change_%"].apply(lambda v: f"{v:+.2f}%" if v is not None else "—")
    display["pct_of_acct"]   = display["pct_of_acct"].apply(lambda v: f"{v:.1f}%" if v is not None else "—")
    display.columns = ["Symbol", "Name", "Shares", "Price", "Mkt Value", "Day $",
                       "Day %", "Cost Basis", "Gain $", "Gain %", "% Acct", "Type"]
    st.table(display.set_index("Symbol"))

    # --- Strategy: goals, watchlist, theses ---
    # All three are stored together in ~/.stock_dashboard/strategy/<owner>.json
    # and fed into the AI review + chat below.
    strategy = load_strategy(owner_key)
    profile_for_goal = st.session_state.get("profile") or load_profile()

    with st.expander("🎯 Goal tracker", expanded=False):
        st.caption(
            "Project this portfolio forward and check whether you're on track. "
            "Monthly contribution comes from your investor profile."
        )
        g = strategy.get("goal") or {}
        gc1, gc2, gc3 = st.columns(3)
        target = gc1.number_input(
            "Target value ($)", min_value=0, max_value=100_000_000, step=10_000,
            value=int(g.get("target", 1_000_000)),
            key=f"goal_target_{owner_key}",
        )
        years = gc2.number_input(
            "Years from now", min_value=1, max_value=80,
            value=int(g.get("years", profile_for_goal.get("horizon_years") or 20)),
            key=f"goal_years_{owner_key}",
        )
        annual_return = gc3.number_input(
            "Assumed annual return (%)", min_value=-20.0, max_value=30.0, step=0.5,
            value=float(g.get("annual_return_pct", 7.0)),
            key=f"goal_return_{owner_key}",
            help="Long-run S&P 500 has averaged ~7% real / ~10% nominal. Be honest, not optimistic.",
        )
        monthly_contrib = float(profile_for_goal.get("monthly_contrib") or 0)
        if monthly_contrib == 0:
            st.caption("⚠️ Monthly contribution is $0 in your profile — projection assumes no new deposits.")

        if st.button("Save goal", key=f"save_goal_{owner_key}"):
            strategy["goal"] = {
                "target": target, "years": years, "annual_return_pct": annual_return,
            }
            save_strategy(owner_key, strategy)
            invalidate_ai_cache(sess_review_key)
            st.success("Goal saved.")
            st.rerun()

        proj = project_goal_progress(
            current=float(total_mv), monthly_contrib=monthly_contrib,
            target=float(target), years=int(years), annual_return_pct=float(annual_return),
        )
        if proj:
            verdict = "✅ On track" if proj["on_track"] else "⚠️ Off track"
            gap = proj["gap"]
            gap_str = f"${gap:+,.0f} vs target"
            mt1, mt2, mt3 = st.columns(3)
            mt1.metric("Projected final value", f"${proj['final']:,.0f}", gap_str)
            mt2.metric("Status", verdict)
            mt3.metric("Need / month to hit target", f"${proj['needed_per_month']:,.0f}")

            series_df = pd.DataFrame(proj["series"])
            series_df["year"] = series_df["month"] / 12
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=series_df["year"], y=series_df["value"],
                mode="lines", name="Projected",
                line=dict(width=2),
            ))
            fig_g.add_hline(
                y=target, line_dash="dash", line_color="gray",
                annotation_text=f"Target ${target:,.0f}", annotation_position="top right",
            )
            fig_g.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Years from now", yaxis_title="Portfolio value ($)",
                showlegend=False,
            )
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

    with st.expander("👀 Watchlist", expanded=False):
        st.caption(
            "Stocks you're considering but don't own yet. The AI review will know "
            "about them and can suggest if/when one fits your portfolio."
        )
        watchlist = list(strategy.get("watchlist") or [])
        if watchlist:
            for i, item in enumerate(watchlist):
                wc1, wc2, wc3 = st.columns([2, 5, 1])
                wc1.markdown(f"**{item.get('symbol', '?')}**")
                wc2.caption(item.get("note") or "—")
                if wc3.button("🗑", key=f"del_watch_{owner_key}_{i}"):
                    watchlist.pop(i)
                    strategy["watchlist"] = watchlist
                    save_strategy(owner_key, strategy)
                    invalidate_ai_cache(sess_review_key)
                    st.rerun()
        else:
            st.caption("Empty. Add a ticker below.")

        with st.form(f"watch_form_{owner_key}", clear_on_submit=True):
            wac1, wac2, wac3 = st.columns([2, 5, 1])
            new_sym = wac1.text_input("Ticker", placeholder="e.g. NVDA",
                                      label_visibility="collapsed").strip().upper()
            new_note = wac2.text_input("Why you're watching it",
                                       placeholder="e.g. waiting for a pullback under $130",
                                       label_visibility="collapsed")
            add_clicked = wac3.form_submit_button("Add")
            if add_clicked and new_sym:
                # Dedupe — replace existing entry for that symbol if present
                watchlist = [w for w in watchlist if w.get("symbol") != new_sym]
                watchlist.append({"symbol": new_sym, "note": new_note.strip()})
                strategy["watchlist"] = watchlist
                save_strategy(owner_key, strategy)
                invalidate_ai_cache(sess_review_key)
                st.rerun()

    with st.expander("📝 Theses (why you own each)", expanded=False):
        st.caption(
            "One-line reason for each holding. Forces you to articulate the case, "
            "and gives the AI review real context instead of guessing from the ticker."
        )
        theses = dict(strategy.get("theses") or {})
        with st.form(f"theses_form_{owner_key}"):
            edited: dict[str, str] = {}
            for sym in holdings["symbol"]:
                edited[sym] = st.text_input(
                    sym, value=theses.get(sym, ""),
                    placeholder="e.g. cash machine, buying back shares aggressively",
                    key=f"thesis_{owner_key}_{sym}",
                )
            if st.form_submit_button("Save theses", use_container_width=True):
                strategy["theses"] = {k: v.strip() for k, v in edited.items() if v.strip()}
                save_strategy(owner_key, strategy)
                invalidate_ai_cache(sess_review_key)
                st.success("Theses saved.")
                st.rerun()

    st.divider()

    # Allocation pies
    st.subheader("Allocation")
    acol1, acol2 = st.columns(2)
    with acol1:
        st.caption("By holding")
        fig_h = go.Figure(data=[go.Pie(
            labels=holdings["symbol"], values=holdings["market_value"],
            hole=0.4, textinfo="label+percent",
        )])
        fig_h.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

    with st.spinner("Looking up sectors…"):
        sectors = fetch_sectors(tuple(holdings["symbol"].tolist()))

    with acol2:
        st.caption("By sector")
        sec_df = holdings.assign(sector=holdings["symbol"].map(sectors)) \
            .groupby("sector", as_index=False)["market_value"].sum()
        fig_s = go.Figure(data=[go.Pie(
            labels=sec_df["sector"], values=sec_df["market_value"],
            hole=0.4, textinfo="label+percent",
        )])
        fig_s.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # AI review
    st.subheader(f"🤖 AI portfolio review · {owner_label}")
    st.caption("✨ AI generated · using full holdings data (no account number)")
    rcol, rbtn = st.columns([5, 1])
    with rbtn:
        refresh_p = st.button("Refresh", key=f"refresh_review_{owner_key}", use_container_width=True)
    if refresh_p:
        invalidate_ai_cache(sess_review_key)
    with st.spinner("Claude is reviewing this portfolio…"):
        review_text = get_or_cache_ai(
            sess_review_key,
            lambda: ai_portfolio_review(
                holdings, p_totals, cash, sectors,
                profile=st.session_state.get("profile") or load_profile(),
                strategy=load_strategy(owner_key),
            ),
        )
    with rcol:
        st.markdown(review_text.replace("$", "\\$"))

    # Chat
    st.divider()
    st.subheader("💬 Ask follow-up questions")
    st.caption(f"✨ AI generated · context: {owner_label}")

    if sess_chat_key not in st.session_state:
        st.session_state[sess_chat_key] = load_chat(f"portfolio_{owner_key}")

    if not st.session_state[sess_chat_key]:
        st.caption("Try:")
        samples = [
            "What's my biggest single risk right now?",
            "How would I diversify without a big tax hit?",
            "How does this compare to an S&P 500 portfolio?",
        ]
        scols = st.columns(len(samples))
        for i, (col, s) in enumerate(zip(scols, samples)):
            if col.button(s, key=f"sample_{owner_key}_{i}", use_container_width=True):
                st.session_state[f"_pending_chat_{owner_key}"] = s
                st.rerun()

    for msg in st.session_state[sess_chat_key]:
        icon = "🧑" if msg["role"] == "user" else "🤖"
        label = "You" if msg["role"] == "user" else "Claude"
        st.markdown(f"**{icon} {label}**")
        st.markdown(msg["content"].replace("$", "\\$"))
        st.markdown("---")

    with st.form(f"chat_form_{owner_key}", clear_on_submit=True):
        p_q = st.text_input(
            "Ask a question",
            placeholder=f"e.g. should {owner_label.lower().replace('s portfolio','').strip("'s ")} trim the biggest position?",
            label_visibility="collapsed",
        )
        p_submit = st.form_submit_button("Send", use_container_width=True)

    p_pending = st.session_state.pop(f"_pending_chat_{owner_key}", None)
    p_question = (p_q.strip() if p_submit and p_q else None) or p_pending

    if p_question:
        st.session_state[sess_chat_key].append({"role": "user", "content": p_question})
        p_system = build_portfolio_chat_context(
            holdings, p_totals, cash, sectors,
            review=st.session_state.get(sess_review_key),
            profile=st.session_state.get("profile") or load_profile(),
            strategy=load_strategy(owner_key),
        )
        # Tell Claude WHOSE portfolio this is so advice is appropriately framed
        p_system = (
            f"NOTE: This is {owner_label} (managed by the user). When giving advice, "
            f"frame it as advice for {'the user themselves' if owner_key == 'mine' else 'a family member the user is managing money for'}. "
            f"\n\n" + p_system
        )
        p_history = st.session_state[sess_chat_key][-20:]
        with st.spinner("Thinking…"):
            p_answer = call_claude_messages(p_history, max_tokens=700, system=p_system)
        st.session_state[sess_chat_key].append({"role": "assistant", "content": p_answer})
        save_chat(f"portfolio_{owner_key}", st.session_state[sess_chat_key])
        st.rerun()

    if st.session_state[sess_chat_key]:
        if st.button("Clear chat", key=f"clear_chat_{owner_key}"):
            st.session_state[sess_chat_key] = []
            save_chat(f"portfolio_{owner_key}", [])
            st.rerun()

    # Manage
    st.divider()
    with st.expander(f"⚙️ Manage {owner_label}"):
        # "mine" is reserved as the user's own portfolio — only the second tab
        # can be renamed (e.g. for a partner, parent, or shared account).
        if owner_key != "mine":
            with st.form(f"rename_form_{owner_key}", clear_on_submit=False):
                new_label = st.text_input(
                    "Portfolio name",
                    value=owner_label,
                    help="Shown on the tab and throughout this portfolio's UI. "
                         "Saved to settings.json.",
                )
                if st.form_submit_button("Save name"):
                    new_label = new_label.strip()
                    if not new_label:
                        st.error("Name can't be empty.")
                    elif new_label != owner_label:
                        settings = load_settings()
                        labels = dict(settings.get("owner_labels") or {})
                        labels[owner_key] = new_label
                        settings["owner_labels"] = labels
                        save_settings(settings)
                        # Invalidate any cached AI output that referenced the old label
                        invalidate_ai_cache(sess_review_key)
                        st.session_state.pop(sess_chat_key, None)
                        st.success(f"Renamed to “{new_label}”.")
                        st.rerun()
            st.divider()

        confirm = st.checkbox(
            f"Yes, I want to delete the saved CSV for {owner_label}",
            key=f"confirm_del_{owner_key}",
        )
        if st.button(f"🗑 Delete saved CSV", key=f"del_{owner_key}", disabled=not confirm):
            try:
                csv_path.unlink(missing_ok=True)
            except Exception as e:
                st.error(f"Couldn't delete file: {e}")
            st.session_state.pop(sess_data_key, None)
            invalidate_ai_cache(sess_review_key)
            st.session_state.pop(sess_chat_key, None)
            save_chat(f"portfolio_{owner_key}", [])  # also delete persisted chat
            st.rerun()


with tab_mine:
    render_portfolio_tab("mine", mine_label)

with tab_moms:
    render_portfolio_tab("moms", moms_label)


with tab_market:
    # ============================================================
    # Market Today tab — aggregated news + AI briefing + earnings + chat
    # ============================================================
    st.subheader("🌍 Market Today")
    st.caption(
        "Aggregated from CNBC, MarketWatch, Yahoo Finance, WSJ, and Finnhub. "
        "Refreshed every 5 minutes. Chat below to ask follow-ups."
    )

    # Portfolio tickers (if uploaded) so we can cross-reference
    portfolio_tickers = ()
    if "portfolio" in st.session_state:
        portfolio_tickers = tuple(st.session_state.portfolio["holdings"]["symbol"].tolist())

    with st.spinner("Pulling headlines from 6 sources…"):
        market_news, news_errors = fetch_market_news()
        earnings_week = fetch_earnings_week(portfolio_tickers)

    if news_errors:
        with st.expander(f"⚠️ {len(news_errors)} source(s) had issues"):
            for src, msg in news_errors.items():
                st.code(f"{src}: {msg}")

    if not market_news:
        st.error("Couldn't load any market news. Check internet / API keys.")
        st.stop()

    # --- AI briefing ---
    st.subheader("🤖 Daily market briefing")
    st.caption(f"✨ AI generated · from {len(market_news)} headlines across {len(RSS_FEEDS)+1} sources")
    mcol1, mcol2 = st.columns([5, 1])
    with mcol2:
        refresh_m = st.button("Refresh", key="refresh_market", use_container_width=True)

    cache_m = "market_briefing"
    if refresh_m:
        invalidate_ai_cache(cache_m)
    with st.spinner("Claude is reading the news…"):
        market_text = get_or_cache_ai(
            cache_m,
            lambda: ai_market_briefing(
                market_news, earnings_week, portfolio_tickers,
            ),
        )
    with mcol1:
        st.markdown(market_text.replace("$", "\\$"))

    st.divider()

    # --- Earnings this week ---
    if earnings_week:
        st.subheader(f"📅 Earnings this week ({len(earnings_week)})")
        owned_set = set(portfolio_tickers)
        rows = []
        for e in earnings_week[:30]:
            sym = e.get("symbol", "")
            owned = "⭐" if sym in owned_set else ""
            rows.append({
                "Date":   e.get("date", ""),
                "Symbol": f"{owned} {sym}".strip(),
                "When":   e.get("hour", "").upper() or "—",
                "EPS Est": f"${e['epsEstimate']:.2f}" if e.get("epsEstimate") else "—",
                "Rev Est": fmt_big(e["revenueEstimate"]) if e.get("revenueEstimate") else "—",
            })
        st.table(pd.DataFrame(rows).set_index("Date"))
        st.caption("⭐ = in your portfolio")
        st.divider()

    # --- News list ---
    st.subheader(f"📰 Latest headlines ({len(market_news)})")
    src_filter = st.multiselect(
        "Filter by source",
        sorted({n["source"] for n in market_news}),
        default=[],
        help="Leave empty to show all.",
    )
    visible = [n for n in market_news if not src_filter or n["source"] in src_filter]

    for n in visible[:50]:
        when = datetime.fromtimestamp(n["ts"]).strftime("%Y-%m-%d %H:%M") if n["ts"] else ""
        st.markdown(f"**[{n['title']}]({n['url']})**  \n_{n['source']} · {when}_")
        if n["summary"]:
            st.caption(n["summary"])
        st.markdown("")

    st.divider()

    # --- Market chat ---
    st.subheader("💬 Ask about today's market")
    st.caption("✨ AI generated · knows all headlines above and your portfolio")

    if "market_chat_history" not in st.session_state:
        st.session_state.market_chat_history = load_chat("market")

    if not st.session_state.market_chat_history:
        st.caption("Try:")
        samples_m = [
            "What's the single most important story today?",
            "Anything in the news about my portfolio holdings?",
            "What sectors are getting hit?",
        ]
        scols_m = st.columns(len(samples_m))
        for i, (col, s) in enumerate(zip(scols_m, samples_m)):
            if col.button(s, key=f"m_sample_{i}", use_container_width=True):
                st.session_state._pending_m_chat = s
                st.rerun()

    for msg in st.session_state.market_chat_history:
        icon = "🧑" if msg["role"] == "user" else "🤖"
        label = "You" if msg["role"] == "user" else "Claude"
        st.markdown(f"**{icon} {label}**")
        st.markdown(msg["content"].replace("$", "\\$"))
        st.markdown("---")

    with st.form("market_chat_form", clear_on_submit=True):
        m_q = st.text_input(
            "Ask a question",
            placeholder="e.g. is there any news that affects my portfolio?",
            label_visibility="collapsed",
        )
        m_submit = st.form_submit_button("Send", use_container_width=True)

    m_pending = st.session_state.pop("_pending_m_chat", None)
    m_question = (m_q.strip() if m_submit and m_q else None) or m_pending

    if m_question:
        st.session_state.market_chat_history.append({"role": "user", "content": m_question})
        m_system = build_market_chat_context(
            market_news, earnings_week,
            briefing=st.session_state.get(cache_m),
            portfolio_tickers=portfolio_tickers,
            profile=st.session_state.get("profile"),
        )
        m_history = st.session_state.market_chat_history[-20:]
        with st.spinner("Thinking…"):
            m_answer = call_claude_messages(m_history, max_tokens=700, system=m_system)
        st.session_state.market_chat_history.append({"role": "assistant", "content": m_answer})
        save_chat("market", st.session_state.market_chat_history)
        st.rerun()

    if st.session_state.market_chat_history:
        if st.button("Clear chat", key="clear_market_chat"):
            st.session_state.market_chat_history = []
            save_chat("market", [])
            st.rerun()


# ============================================================
# Sidebar chatbot — context-aware Q&A about what's on the page
# ============================================================

st.sidebar.divider()
st.sidebar.subheader(f"💬 Ask about {ticker}")
st.sidebar.caption("✨ AI generated · context-aware")

# When the ticker changes, load (or initialize) that ticker's persistent chat.
# This way switching back to AAPL later restores the prior conversation.
if st.session_state.get("chat_ticker") != ticker:
    st.session_state.chat_ticker = ticker
    st.session_state.chat_history = load_chat(f"stock_{ticker}")
    prune_stock_chats(max_keep=10)

# Sample prompts to get them started
if not st.session_state.chat_history:
    st.sidebar.caption("Try:")
    samples = [
        f"Why does the RSI matter here?",
        f"Is {ticker} expensive right now?",
        f"What should I watch this week?",
    ]
    for i, s in enumerate(samples):
        if st.sidebar.button(s, key=f"sample_{i}", use_container_width=True):
            st.session_state._pending_chat = s
            st.rerun()

# Display history (oldest first)
for msg in st.session_state.chat_history:
    icon = "🧑" if msg["role"] == "user" else "🤖"
    label = "You" if msg["role"] == "user" else "Claude"
    st.sidebar.markdown(f"**{icon} {label}**")
    st.sidebar.markdown(msg["content"].replace("$", "\\$"))
    st.sidebar.markdown("---")

# Input — use a form so Enter submits and the field clears on rerun
with st.sidebar.form("chat_form", clear_on_submit=True):
    user_q = st.text_input("Ask a question", placeholder=f"e.g. why is RSI high for {ticker}?", label_visibility="collapsed")
    submitted = st.form_submit_button("Send", use_container_width=True)

pending = st.session_state.pop("_pending_chat", None)
question = (user_q.strip() if submitted and user_q else None) or pending

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    system = build_chat_context(
        ticker, profile, quote, metrics, ind, news,
        briefing=st.session_state.get(f"brief_{ticker}"),
        ta=st.session_state.get(f"ta_{ticker}_{period}"),
    )
    # Cap history at last 10 exchanges so we don't blow the context budget
    history = st.session_state.chat_history[-20:]
    with st.sidebar:
        with st.spinner("Thinking…"):
            answer = call_claude_messages(history, max_tokens=600, system=system)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat(f"stock_{ticker}", st.session_state.chat_history)
    st.rerun()

if st.session_state.chat_history:
    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        save_chat(f"stock_{ticker}", [])
        st.rerun()
