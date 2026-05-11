"""
Streamlit UI for the trading bot.

Run with:
    pip install streamlit
    streamlit run app.py

The bot runs in a background thread. The UI just reads its shared state
and shows Start / Pause / Stop buttons.
"""

from threading import Event, Thread
import time

import streamlit as st

import trading_bot as bot

st.set_page_config(page_title="Briefing Markets — Bot", page_icon="📈", layout="wide")


# ============================================================
# Session state — survives across Streamlit reruns
# ============================================================

if "bot_state" not in st.session_state:
    st.session_state.bot_state = bot.BotState()
    st.session_state.stop_event = Event()
    st.session_state.pause_event = Event()
    st.session_state.pause_event.set()  # start paused
    st.session_state.thread = None


def ensure_thread_running():
    """Start the background thread once. Idempotent."""
    t = st.session_state.thread
    if t is None or not t.is_alive():
        st.session_state.stop_event.clear()
        thread = Thread(
            target=bot.run_loop,
            args=(
                st.session_state.bot_state,
                st.session_state.stop_event,
                st.session_state.pause_event,
            ),
            daemon=True,
        )
        thread.start()
        st.session_state.thread = thread


# ============================================================
# Header + controls
# ============================================================

st.title("Briefing Markets — Paper Trading Bot")
st.caption(
    f"Strategy: buy {bot.STOCK} when {bot.SHORT_WINDOW}-min avg crosses above "
    f"{bot.LONG_WINDOW}-min avg. Paper money only."
)

paused = st.session_state.pause_event.is_set()
status = "Paused" if paused else "Running"
st.markdown(f"**Status:** {status}")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Start", use_container_width=True, disabled=not paused):
        ensure_thread_running()
        st.session_state.pause_event.clear()
        st.rerun()
with c2:
    if st.button("Pause", use_container_width=True, disabled=paused):
        st.session_state.pause_event.set()
        st.rerun()
with c3:
    if st.button("Run one tick now", use_container_width=True):
        # Useful when paused or for testing — runs synchronously.
        bot.do_one_tick(st.session_state.bot_state)
        st.rerun()

st.divider()

# ============================================================
# Live metrics
# ============================================================

state = st.session_state.bot_state
with state.lock:
    price = state.price
    short_avg = state.short_avg
    long_avg = state.long_avg
    position_qty = state.position_qty
    last_action = state.last_action
    last_update = state.last_update
    log_lines = list(state.log)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${price:.2f}" if price is not None else "—")
m2.metric("Short avg", f"${short_avg:.2f}" if short_avg is not None else "—")
m3.metric("Long avg", f"${long_avg:.2f}" if long_avg is not None else "—")
m4.metric("Position", f"{position_qty} shares")

if short_avg is not None and long_avg is not None:
    if short_avg > long_avg:
        st.info("Short avg is **above** long avg → trend is up (bot wants to be long).")
    else:
        st.info("Short avg is **below** long avg → trend is down (bot wants to be flat).")

st.markdown(f"**Last action:** {last_action}")
if last_update:
    st.caption(f"Last update: {last_update.strftime('%H:%M:%S')}")

st.divider()

# ============================================================
# Log
# ============================================================

st.subheader("Activity log")
if log_lines:
    st.code("\n".join(reversed(log_lines)), language=None)
else:
    st.caption("No activity yet. Click Start.")

# ============================================================
# Auto-refresh while running so metrics stay live.
# ============================================================

if not paused:
    time.sleep(2)
    st.rerun()
