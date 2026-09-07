AutoNSE AI v4 — Chart-first NSE research scanner

Features
- NSE universe up to 1000 stocks
- Angel One SmartAPI mode when Streamlit secrets are configured
- Research fallback using Yahoo Finance
- Directional BUY / SELL / WAIT / WATCH logic
- Pivot + EMA support/resistance
- Structure-based entry, stop loss and targets
- R:R validation; does not force a trade
- Interactive candlestick chart with EMA20/50/200
- Chart annotations for support, resistance, entry, SL, T1 and T2
- Reason text explaining why the setup passed/failed

IMPORTANT
- Do not paste API key, password or TOTP secret into chat.
- Put them in Streamlit Cloud Secrets.
- Suggested keys:
  ANGEL_API_KEY="..."
  ANGEL_CLIENT_CODE="..."
  ANGEL_PASSWORD="..."
  ANGEL_TOTP_SECRET="..."

The app is research software, not an execution bot. Scores are heuristics, not calibrated probabilities.
