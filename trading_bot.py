"""
Simple Paper Trading Bot - Moving Average Crossover Strategy

Uses Alpaca's paper trading API (fake money, real market data) via the
modern `alpaca-py` SDK.

SETUP:
1. Sign up at https://alpaca.markets (free)
2. Go to your dashboard > Paper Trading > API Keys
3. Put your keys in a `.env` file next to this script:
       ALPACA_API_KEY=...
       ALPACA_API_SECRET=...
4. Install dependencies: pip install alpaca-py pandas python-dotenv

HOW IT WORKS:
- Every minute, it checks the stock price
- If the short-term average crosses ABOVE the long-term average -> BUY
- If the short-term average crosses BELOW the long-term average -> SELL
- This is called a "Moving Average Crossover" strategy

This file can be run directly (`python trading_bot.py`) for a CLI loop, or
imported by `app.py` for the Streamlit UI.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, Lock
from typing import Optional
import os
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER      = True

STOCK        = "AAPL"
SHORT_WINDOW = 5
LONG_WINDOW  = 20
QTY          = 1
SLEEP_SEC    = 60

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "Missing ALPACA_API_KEY or ALPACA_API_SECRET. "
        "Add them to a .env file next to this script."
    )

# ============================================================
# Alpaca clients
# ============================================================

trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, API_SECRET)


def get_prices(symbol, limit=30):
    """Fetch recent 1-minute closing prices."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=limit * 4)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(request).df

    if "symbol" in bars.index.names:
        bars = bars.xs(symbol, level="symbol")

    return bars["close"].tail(limit)


def get_position(symbol):
    """Check if we currently own the stock. Returns share count, or 0."""
    try:
        pos = trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def submit_market_order(symbol, qty, side):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
    )
    trading_client.submit_order(order)


# ============================================================
# Shared state — the "whiteboard" between bot and UI
# ============================================================

@dataclass
class BotState:
    """Snapshot of what the bot last saw / did. Read by the UI."""
    price: Optional[float] = None
    short_avg: Optional[float] = None
    long_avg: Optional[float] = None
    position_qty: int = 0
    last_action: str = "—"
    last_update: Optional[datetime] = None
    log: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def add_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.log.append(f"[{ts}] {msg}")
            # cap log length so it doesn't grow forever
            if len(self.log) > 200:
                self.log = self.log[-200:]


def do_one_tick(state: BotState):
    """Run a single iteration of the strategy and update shared state."""
    try:
        prices = get_prices(STOCK, limit=LONG_WINDOW + 5)

        short_avg = float(prices[-SHORT_WINDOW:].mean())
        long_avg  = float(prices[-LONG_WINDOW:].mean())
        current_price = float(prices.iloc[-1])
        position_qty  = get_position(STOCK)

        action = "No action"

        if short_avg > long_avg and position_qty == 0:
            submit_market_order(STOCK, QTY, OrderSide.BUY)
            action = f"BUY {QTY} share(s) of {STOCK}"
        elif short_avg < long_avg and position_qty > 0:
            submit_market_order(STOCK, position_qty, OrderSide.SELL)
            action = f"SELL {position_qty} share(s) of {STOCK}"

        with state.lock:
            state.price = current_price
            state.short_avg = short_avg
            state.long_avg = long_avg
            state.position_qty = position_qty
            state.last_action = action
            state.last_update = datetime.now()

        state.add_log(
            f"Price ${current_price:.2f} | short ${short_avg:.2f} | "
            f"long ${long_avg:.2f} | pos {position_qty} | {action}"
        )

    except Exception as e:
        state.add_log(f"Error: {e}")


def run_loop(state: BotState, stop_event: Event, pause_event: Event):
    """Background loop. Honors pause_event (set = paused) and stop_event."""
    state.add_log(
        f"Bot started. {STOCK} | short={SHORT_WINDOW} long={LONG_WINDOW} "
        f"qty={QTY} every {SLEEP_SEC}s"
    )
    while not stop_event.is_set():
        if not pause_event.is_set():
            do_one_tick(state)
        # Wait in small slices so pause/stop respond quickly.
        for _ in range(SLEEP_SEC):
            if stop_event.is_set():
                return
            time.sleep(1)


def run_cli():
    """Standalone CLI runner — prints to stdout, no UI."""
    state = BotState()
    stop_event = Event()
    pause_event = Event()  # never set; CLI has no pause
    print(f"Bot started. Trading {STOCK} with {'paper' if PAPER else 'LIVE'} money.")
    print(f"Strategy: Buy when {SHORT_WINDOW}-bar avg crosses above {LONG_WINDOW}-bar avg")
    print("Ctrl+C to stop.\n")
    try:
        while True:
            do_one_tick(state)
            # Print the latest log line we just produced.
            if state.log:
                print(state.log[-1])
            time.sleep(SLEEP_SEC)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    run_cli()
