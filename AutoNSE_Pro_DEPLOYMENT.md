# AutoNSE Pro Scanner

NSE-only Streamlit scanner for Intraday and Swing setups.

## Local

```bash
python -m pip install -r AutoNSE_Pro_requirements.txt
streamlit run AutoNSE_Pro_app.py
```

## Permanent website

Put `app.py` and `requirements.txt` in a GitHub repository, then deploy the repository with Streamlit Community Cloud. After deployment, use the generated `*.streamlit.app` URL from any phone or computer; you do not need to run Streamlit manually on your own PC.

## Data note

This version uses yfinance for market data. It is suitable for screening/prototyping but is not an exchange-grade live trading feed. For broker-grade intraday data, replace the data layer with Angel One SmartAPI.

## Risk note

Win Probability is a rule-based score, not a statistically calibrated probability and never guarantees profit.
