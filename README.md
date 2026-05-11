# Briefing Markets

A personal stock dashboard and paper-trading bot, built with Streamlit. Track holdings, read AI-generated briefings on the market and your portfolio, and (optionally) let a moving-average crossover bot place paper trades against Alpaca.

> Not financial advice. This is a personal tool — verify everything before acting on it.

## Features

**Dashboard (`dashboard.py`)**
- 📊 **Single-stock view** — interactive price chart with moving averages, RSI, MACD, Bollinger bands; quote, fundamentals, recent news, and an AI technical-analysis blurb.
- 💼 **Portfolio tabs** — upload a Schwab Positions CSV per portfolio; holdings table, allocation pies, sector breakdown. Account numbers are stripped on import.
- 🎯 **Strategy layer** per portfolio — set a savings goal with projection chart, maintain a watchlist of names you don't own yet, and write one-line theses for each holding. All three feed into the AI review.
- 🤖 **AI portfolio review + chat** — concentration/sector risk analysis tailored to your investor profile (age, horizon, risk tolerance, target allocation). Follow-up chat with full portfolio context.
- 🌍 **Market Today** — aggregated headlines from CNBC, MarketWatch, Yahoo Finance et al., upcoming earnings, AI market briefing, and a market-aware chat.

**Paper-trading bot (`trading_bot.py` + `app.py`)**
- Simple moving-average crossover strategy on AAPL, running against Alpaca's paper-trading API.
- Streamlit UI with Start / Pause / Run-one-tick controls and a live activity log.

## Quick start

```bash
git clone https://github.com/vkiragi/briefing-markets.git
cd briefing-markets
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your API keys
streamlit run dashboard.py
```

Run the paper-trading bot UI separately with `streamlit run app.py`.

## Configuration

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
FINNHUB_API_KEY=...
OPENROUTER_API_KEY=...
```

| Variable | Where to get it | Used for |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | [alpaca.markets](https://alpaca.markets) → Paper Trading → API Keys | Historical price bars (IEX feed), paper-trading bot |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io) (free tier works) | Real-time quotes, company profile, fundamentals, news |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | AI briefings via Claude Haiku 4.5 |

## Data storage

All personal state lives in `~/.stock_dashboard/` — never committed to the repo:

```
~/.stock_dashboard/
├── profile.json          # your investor profile (age, horizon, risk, targets)
├── settings.json         # UI settings (custom tab labels)
├── portfolios/           # your uploaded Positions CSVs
│   ├── mine.csv
│   └── moms.csv          # second tab — rename from the UI
├── strategy/             # goals, watchlist, theses per portfolio
│   └── mine.json
├── chats/                # persisted chat histories
└── ai_cache/             # cached AI briefings/reviews — survives reloads.
                          # Click "Refresh" on any AI block to regenerate.
```

## Renaming the second portfolio

The two portfolio tabs default to **"My Portfolio"** and **"Mom's Portfolio"**. To rename the second tab (e.g. to "Partner", "Dad", "Joint Account"), open the tab → **⚙️ Manage** expander → enter a new name.

## Tech stack

- **UI:** Streamlit, Plotly
- **Data:** Alpaca (price history), Finnhub (quotes / fundamentals / news), feedparser (RSS)
- **AI:** OpenRouter → Claude Haiku 4.5
- **Storage:** Local JSON / CSV under `~/.stock_dashboard/`

## License

[MIT](LICENSE)
