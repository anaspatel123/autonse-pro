
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# AUTO NSE MOMENTUM SCANNER
# NSE CASH EQUITY ONLY
# ============================================================

st.set_page_config(
    page_title="Auto NSE Momentum Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# 45 highly liquid NSE stocks. Deliberately NO .BO / BSE symbols.
# ----------------------------------------------------------------
NSE_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "TATAMOTORS.NS", "AXISBANK.NS", "MARUTI.NS", "ITC.NS",
    "BHARTIARTL.NS", "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SUNPHARMA.NS", "TITAN.NS",
    "M&M.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "NESTLEIND.NS", "TATACONSUM.NS", "HINDALCO.NS", "GRASIM.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "TVSMOTOR.NS", "BEL.NS", "HAL.NS", "TRENT.NS", "INDIGO.NS",
]

INDEX = "^NSEI"  # Nifty 50
INTRADAY_GAINER_THRESHOLD = 1.5
INTRADAY_LOSER_THRESHOLD = -1.5
INTRADAY_RSI_BUY = 60
INTRADAY_RSI_SELL = 40
VOLUME_MULTIPLIER = 2.0

# ============================================================
# HELPERS
# ============================================================

def clean_ticker(ticker: str) -> str:
    return ticker.replace(".NS", "")

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def fmt_num(value, digits=2):
    if value is None or not np.isfinite(safe_float(value)):
        return "-"
    return f"{float(value):.{digits}f}"

def pct(value):
    return f"{value:.2f}%" if np.isfinite(safe_float(value)) else "-"

def download_data(tickers, period="1y", interval="1d"):
    """
    Download multiple NSE tickers in one Yahoo Finance request.
    Returns a dict: ticker -> OHLCV DataFrame.
    """
    if not tickers:
        return {}

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            prepost=False,
        )
    except Exception as exc:
        st.warning(f"Yahoo Finance download failed: {exc}")
        return {}

    result = {}

    if raw is None or raw.empty:
        return result

    # Multi-ticker download usually gives MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))

        # yfinance can place ticker at either level depending on version.
        if any(t in level0 for t in tickers):
            for ticker in tickers:
                if ticker not in level0:
                    continue
                df = raw[ticker].copy()
                if not df.empty:
                    result[ticker] = normalize_ohlcv(df)
        else:
            for ticker in tickers:
                if ticker not in level1:
                    continue
                df = raw.xs(ticker, axis=1, level=1).copy()
                if not df.empty:
                    result[ticker] = normalize_ohlcv(df)
    else:
        # Single ticker.
        ticker = tickers[0]
        result[ticker] = normalize_ohlcv(raw.copy())

    return result

def normalize_ohlcv(df):
    df = df.copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[needed].copy()
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)

    # Remove duplicate timestamps.
    df = df[~df.index.duplicated(keep="last")]

    return df

def rsi(series, length=14):
    out = ta.rsi(series, length=length)
    return out

def add_daily_indicators(df):
    d = df.copy()
    d["EMA50"] = ta.ema(d["Close"], length=50)
    d["EMA200"] = ta.ema(d["Close"], length=200)
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)
    d["VOL20"] = d["Volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["Volume"] / d["VOL20"].replace(0, np.nan)
    return d

def add_intraday_indicators(df):
    d = df.copy()
    d["EMA20"] = ta.ema(d["Close"], length=20)
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = ta.atr(d["High"], d["Low"], d["Close"], length=14)
    d["VOL20"] = d["Volume"].rolling(20).mean()
    d["VOL_RATIO"] = d["Volume"] / d["VOL20"].replace(0, np.nan)

    # pandas-ta VWAP is session-sensitive. Fall back to manual calculation
    # if the installed version does not return a valid series.
    try:
        d["VWAP"] = ta.vwap(
            d["High"], d["Low"], d["Close"], d["Volume"]
        )
    except Exception:
        typical = (d["High"] + d["Low"] + d["Close"]) / 3
        session = d.index.date
        cum_pv = (typical * d["Volume"]).groupby(session).cumsum()
        cum_vol = d["Volume"].groupby(session).cumsum()
        d["VWAP"] = cum_pv / cum_vol.replace(0, np.nan)

    return d

# ============================================================
# NIFTY TREND
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_nifty_trend():
    data = download_data([INDEX], period="6mo", interval="1d")
    df = data.get(INDEX, pd.DataFrame())

    if df.empty or len(df) < 50:
        return {
            "trend": "UNKNOWN",
            "close": np.nan,
            "ema20": np.nan,
            "ema50": np.nan,
        }

    d = df.copy()
    d["EMA20"] = ta.ema(d["Close"], length=20)
    d["EMA50"] = ta.ema(d["Close"], length=50)

    last = d.iloc[-1]
    close = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])

    if close > ema20 > ema50:
        trend = "BULLISH"
    elif close < ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
    }

# ============================================================
# FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental(ticker):
    """
    Yahoo Finance fundamentals. Yahoo may not provide every field for
    every company; missing data is treated as unavailable, not invented.
    """
    try:
        info = yf.Ticker(ticker).info

        roe = safe_float(info.get("returnOnEquity"))
        debt_equity = safe_float(info.get("debtToEquity"))

        # Yahoo's debtToEquity is commonly reported as a percentage-like
        # number (e.g. 80 means 0.80x). Normalize to ratio.
        if np.isfinite(debt_equity) and debt_equity > 10:
            debt_equity = debt_equity / 100.0

        return {
            "roe": roe * 100 if np.isfinite(roe) and abs(roe) <= 2 else roe,
            "debt_equity": debt_equity,
            "pe": safe_float(info.get("trailingPE")),
            "market_cap": safe_float(info.get("marketCap")),
            "profit_margin": safe_float(info.get("profitMargins")),
            "revenue_growth": safe_float(info.get("revenueGrowth")),
            "earnings_growth": safe_float(info.get("earningsGrowth")),
        }
    except Exception:
        return {
            "roe": np.nan,
            "debt_equity": np.nan,
            "pe": np.nan,
            "market_cap": np.nan,
            "profit_margin": np.nan,
            "revenue_growth": np.nan,
            "earnings_growth": np.nan,
        }

# ============================================================
# PROBABILITY / SCORE
# ============================================================

def calculate_probability(
    side,
    volume_confirmed=False,
    rsi_confirmed=False,
    index_confirmed=False,
    fundamentals_confirmed=False,
):
    """
    User-requested point matrix:
      baseline = 50
      +10 volume
      +10 RSI
      +10 Nifty alignment
      +10 fundamentals
      cap = 95

    IMPORTANT:
    This is a heuristic model score, not a statistically calibrated
    probability unless you later backtest/calibrate it on out-of-sample data.
    """
    score = 50
    if volume_confirmed:
        score += 10
    if rsi_confirmed:
        score += 10
    if index_confirmed:
        score += 10
    if fundamentals_confirmed:
        score += 10

    return min(score, 95)

# ============================================================
# RISK / TARGETS
# ============================================================

def calculate_levels(side, entry, atr_value):
    entry = safe_float(entry)
    atr_value = safe_float(atr_value)

    if not np.isfinite(entry):
        return np.nan, np.nan, np.nan, np.nan

    if not np.isfinite(atr_value) or atr_value <= 0:
        atr_value = entry * 0.01

    # Conservative ATR-based levels.
    risk = max(atr_value * 1.2, entry * 0.003)

    if side == "BUY":
        sl = entry - risk
        t1 = entry + risk * 1.5
        t2 = entry + risk * 2.0
        t3 = entry + risk * 3.0
    else:
        sl = entry + risk
        t1 = entry - risk * 1.5
        t2 = entry - risk * 2.0
        t3 = entry - risk * 3.0

    return sl, t1, t2, t3

# ============================================================
# INTRADAY
# ============================================================

def get_intraday_candidates(daily_data):
    """
    Primary filter:
      top daily gainers > +1.5%
      top daily losers < -1.5%

    Returns at most 10 gainers + 10 losers.
    """
    rows = []

    for ticker, df in daily_data.items():
        if df.empty or len(df) < 25:
            continue

        close = safe_float(df["Close"].iloc[-1])
        prev_close = safe_float(df["Close"].iloc[-2])

        if not np.isfinite(close) or not np.isfinite(prev_close) or prev_close == 0:
            continue

        day_change = (close / prev_close - 1) * 100

        rows.append({
            "Ticker": clean_ticker(ticker),
            "ticker_full": ticker,
            "Current Price": close,
            "Day Change %": day_change,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    gainers = (
        df[df["Day Change %"] > INTRADAY_GAINER_THRESHOLD]
        .sort_values("Day Change %", ascending=False)
        .head(10)
    )

    losers = (
        df[df["Day Change %"] < INTRADAY_LOSER_THRESHOLD]
        .sort_values("Day Change %", ascending=True)
        .head(10)
    )

    return pd.concat([gainers, losers], ignore_index=True)

def analyze_intraday_symbol(row, intraday_data, nifty):
    ticker = row["ticker_full"]
    df = intraday_data.get(ticker, pd.DataFrame())

    if df.empty or len(df) < 30:
        return None

    d = add_intraday_indicators(df).dropna(subset=["EMA20", "RSI14", "VOL20"])
    if len(d) < 25:
        return None

    last = d.iloc[-1]
    close = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    rsi14 = safe_float(last["RSI14"])
    vol_ratio = safe_float(last["VOL_RATIO"])
    vwap_value = safe_float(last["VWAP"])
    atr_value = safe_float(last["ATR14"])

    day_change = safe_float(row["Day Change %"])

    # Opening range = first 30 minutes of each trading session.
    try:
        latest_date = d.index[-1].date()
        session = d[d.index.date == latest_date]
        opening = session.between_time("09:15", "09:45")
        if opening.empty:
            opening = session.head(3)
    except Exception:
        opening = d.tail(3)

    or_high = safe_float(opening["High"].max())
    or_low = safe_float(opening["Low"].min())

    is_gainer = day_change > INTRADAY_GAINER_THRESHOLD
    is_loser = day_change < INTRADAY_LOSER_THRESHOLD

    # User's strict confirmation rules.
    buy_conditions = (
        is_gainer
        and close > ema20
        and rsi14 > INTRADAY_RSI_BUY
        and vol_ratio > VOLUME_MULTIPLIER
    )

    sell_conditions = (
        is_loser
        and close < ema20
        and rsi14 < INTRADAY_RSI_SELL
        and vol_ratio > VOLUME_MULTIPLIER
    )

    # Index confirmation:
    # BUY aligns with bullish Nifty; SELL aligns with bearish Nifty.
    index_buy = nifty["trend"] == "BULLISH"
    index_sell = nifty["trend"] == "BEARISH"

    # Fundamentals are supportive rather than required for intraday.
    fund = get_fundamental(ticker)
    roe = fund["roe"]
    de = fund["debt_equity"]
    fund_confirmed = (
        np.isfinite(roe) and roe > 15
        and np.isfinite(de) and de < 1.5
    )

    if buy_conditions:
        side = "Buy"
        signal = "BUY"
        reason = (
            "Top daily gainer (>1.5%) above 20 EMA with RSI >60 "
            "and >2x 20-period volume confirmation."
        )

        # Extra quality detail.
        if close > vwap_value:
            reason += " Price is also above VWAP."
        if close > or_high:
            reason += " Opening-range breakout confirmed."

        probability = calculate_probability(
            "BUY",
            volume_confirmed=vol_ratio > 2,
            rsi_confirmed=rsi14 > 60,
            index_confirmed=index_buy,
            fundamentals_confirmed=fund_confirmed,
        )

        sl, t1, t2, t3 = calculate_levels("BUY", close, atr_value)

    elif sell_conditions:
        side = "Sell"
        signal = "SELL"
        reason = (
            "Top daily loser (<-1.5%) below 20 EMA with RSI <40 "
            "and >2x 20-period volume confirmation."
        )

        if close < vwap_value:
            reason += " Price is also below VWAP."
        if close < or_low:
            reason += " Opening-range breakdown confirmed."

        probability = calculate_probability(
            "SELL",
            volume_confirmed=vol_ratio > 2,
            rsi_confirmed=rsi14 < 40,
            index_confirmed=index_sell,
            fundamentals_confirmed=fund_confirmed,
        )

        sl, t1, t2, t3 = calculate_levels("SELL", close, atr_value)

    else:
        # Candidate did not pass the strict setup.
        return None

    metrics = (
        f"RSI {rsi14:.1f} | EMA20 {close > ema20 and 'Above' or 'Below'} | "
        f"Vol {vol_ratio:.1f}x | VWAP {vwap_value:.2f} | "
        f"Nifty {nifty['trend']} | ROE {fmt_num(roe,1)}% | D/E {fmt_num(de,2)}"
    )

    return {
        "Ticker": clean_ticker(ticker),
        "Signal": signal,
        "Current Price": close,
        "Intraday/Weekly % Change": day_change,
        "Reason for Selection": reason,
        "Win Probability (%)": probability,
        "Core Metrics": metrics,
        "Entry": close,
        "Stop Loss": sl,
        "Target 1": t1,
        "Target 2": t2,
        "Target 3": t3,
        "RSI": rsi14,
        "Volume Ratio": vol_ratio,
    }

def run_intraday_scan():
    # Daily data first: cheap primary momentum filter.
    daily_data = download_data(
        NSE_UNIVERSE,
        period="3mo",
        interval="1d",
    )

    candidates = get_intraday_candidates(daily_data)

    if candidates.empty:
        return pd.DataFrame(), "No NSE stocks currently crossed the ±1.5% daily momentum filter."

    selected = candidates["ticker_full"].tolist()

    # Only fetch 15m data for the top 10 gainers / top 10 losers.
    intraday_data = download_data(
        selected,
        period="5d",
        interval="15m",
    )

    nifty = get_nifty_trend()

    results = []
    for _, row in candidates.iterrows():
        result = analyze_intraday_symbol(row, intraday_data, nifty)
        if result:
            results.append(result)

    if not results:
        return (
            pd.DataFrame(),
            "Momentum candidates were found, but none passed all strict 15-minute confirmation rules."
        )

    out = pd.DataFrame(results)
    out = out.sort_values(
        ["Win Probability (%)", "Intraday/Weekly % Change"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out, f"Nifty trend: {nifty['trend']}"

# ============================================================
# SWING
# ============================================================

def weekly_monthly_return(df):
    if len(df) < 23:
        return np.nan, np.nan

    close = safe_float(df["Close"].iloc[-1])
    week_base = safe_float(df["Close"].iloc[-6])
    month_base = safe_float(df["Close"].iloc[-22])

    week = (close / week_base - 1) * 100 if week_base else np.nan
    month = (close / month_base - 1) * 100 if month_base else np.nan

    return week, month

def analyze_swing_symbol(ticker, df, nifty):
    if df.empty or len(df) < 210:
        return None

    d = add_daily_indicators(df).dropna(
        subset=["EMA50", "EMA200", "RSI14", "ATR14", "VOL20"]
    )

    if len(d) < 210:
        return None

    last = d.iloc[-1]

    close = safe_float(last["Close"])
    ema50 = safe_float(last["EMA50"])
    ema200 = safe_float(last["EMA200"])
    rsi14 = safe_float(last["RSI14"])
    atr_value = safe_float(last["ATR14"])

    week_return, month_return = weekly_monthly_return(d)

    # Strongest 1W/1M momentum is the primary filter.
    momentum_ok = (
        np.isfinite(week_return)
        and np.isfinite(month_return)
        and max(week_return, month_return) > 5
    )

    technical_ok = (
        close > ema50
        and close > ema200
        and 50 <= rsi14 <= 65
    )

    if not momentum_ok or not technical_ok:
        return None

    fund = get_fundamental(ticker)
    roe = fund["roe"]
    de = fund["debt_equity"]

    # User requested strict fundamental guardrail.
    fundamentals_ok = (
        np.isfinite(de) and de < 1.5
        and np.isfinite(roe) and roe > 15
    )

    if not fundamentals_ok:
        return None

    volume_ratio = safe_float(last["VOL_RATIO"])

    # Swing index alignment = bullish Nifty trend.
    index_confirmed = nifty["trend"] == "BULLISH"

    # For swing, "strong RSI momentum" is 55-65.
    rsi_confirmed = 55 <= rsi14 <= 65

    # Volume confirmation: at least 1.25x 20-day average.
    volume_confirmed = np.isfinite(volume_ratio) and volume_ratio >= 1.25

    probability = calculate_probability(
        "BUY",
        volume_confirmed=volume_confirmed,
        rsi_confirmed=rsi_confirmed,
        index_confirmed=index_confirmed,
        fundamentals_confirmed=True,
    )

    # Pullback/near EMA detail.
    distance_from_ema50 = (close / ema50 - 1) * 100

    reason = (
        f"Strong momentum ({week_return:.1f}% 1W / {month_return:.1f}% 1M) "
        f"above 50 & 200 EMA, RSI {rsi14:.1f}, "
        f"with Low Debt ({de:.2f}x) and High ROE ({roe:.1f}%)."
    )

    if abs(distance_from_ema50) <= 5:
        reason += " Price is within 5% of the 50 EMA, supporting a cleaner entry."
    if volume_confirmed:
        reason += " Volume is above its 20-day average."

    sl, t1, t2, t3 = calculate_levels("BUY", close, atr_value)

    metrics = (
        f"1W {week_return:.2f}% | 1M {month_return:.2f}% | "
        f"RSI {rsi14:.1f} | EMA50 {ema50:.2f} | EMA200 {ema200:.2f} | "
        f"Vol {fmt_num(volume_ratio,1)}x | Nifty {nifty['trend']} | "
        f"ROE {roe:.1f}% | D/E {de:.2f}"
    )

    return {
        "Ticker": clean_ticker(ticker),
        "Signal": "BUY",
        "Current Price": close,
        "Intraday/Weekly % Change": week_return,
        "Reason for Selection": reason,
        "Win Probability (%)": probability,
        "Core Metrics": metrics,
        "Entry": close,
        "Stop Loss": sl,
        "Target 1": t1,
        "Target 2": t2,
        "Target 3": t3,
        "RSI": rsi14,
        "Volume Ratio": volume_ratio,
    }

def run_swing_scan():
    daily_data = download_data(
        NSE_UNIVERSE,
        period="2y",
        interval="1d",
    )

    if not daily_data:
        return pd.DataFrame(), "No data returned by Yahoo Finance."

    # Momentum ranking first.
    ranking = []

    for ticker, df in daily_data.items():
        if df.empty or len(df) < 30:
            continue

        week_return, month_return = weekly_monthly_return(df)

        if np.isfinite(week_return) and np.isfinite(month_return):
            ranking.append({
                "ticker": ticker,
                "week": week_return,
                "month": month_return,
                "momentum": max(week_return, month_return),
            })

    if not ranking:
        return pd.DataFrame(), "Unable to calculate momentum ranking."

    ranking_df = (
        pd.DataFrame(ranking)
        .sort_values("momentum", ascending=False)
        .head(20)
    )

    nifty = get_nifty_trend()

    results = []

    # Fundamentals are the slowest part, so only the strongest 20
    # momentum candidates are fully evaluated.
    for ticker in ranking_df["ticker"]:
        result = analyze_swing_symbol(ticker, daily_data[ticker], nifty)
        if result:
            results.append(result)

    if not results:
        return (
            pd.DataFrame(),
            "Strong weekly/monthly momentum stocks were found, but none passed all strict technical + fundamental guardrails."
        )

    out = pd.DataFrame(results)
    out = out.sort_values(
        ["Win Probability (%)", "Intraday/Weekly % Change"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out, f"Nifty trend: {nifty['trend']}"

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📈 Auto NSE Momentum Scanner")
st.caption(
    "NSE cash-equity research scanner • Top Gainers/Losers + Momentum + "
    "Technical + Volume + Fundamentals"
)

with st.sidebar:
    st.header("Scanner Settings")

    mode = st.selectbox(
        "Trading Mode",
        ["Intraday Mode", "Swing Trading Mode"],
    )

    st.markdown("---")

    st.write("**Universe:** 45 liquid NSE stocks")
    st.write("**Exchange:** NSE only")
    st.write("**BSE:** Excluded")
    st.write("**Data:** Yahoo Finance")

    st.markdown("---")

    scan_button = st.button(
        "🔎 Scan Stocks",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.warning(
        "The displayed Win Probability is a rule-based model score, "
        "not a guaranteed or statistically calibrated probability."
    )

# Persistent results.
if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()
    st.session_state.scan_message = ""

if scan_button:
    with st.spinner("Scanning NSE stocks and calculating signals..."):
        started = time.time()

        if mode == "Intraday Mode":
            results, message = run_intraday_scan()
        else:
            results, message = run_swing_scan()

        st.session_state.scan_results = results
        st.session_state.scan_message = message

        elapsed = time.time() - started
        st.session_state.scan_message += f" | Scan time: {elapsed:.1f}s"

results = st.session_state.scan_results

if st.session_state.scan_message:
    st.info(st.session_state.scan_message)

if results.empty:
    st.markdown("### Ready")
    st.write(
        "Select a mode and click **Scan Stocks**. "
        "The app first applies the momentum filter, then performs the stricter confirmation checks."
    )

    st.markdown("#### Intraday rules")
    st.write(
        "BUY: daily gain > 1.5% + 15m price above 20 EMA + RSI > 60 + "
        "volume > 2x 20-period average."
    )
    st.write(
        "SELL: daily loss < -1.5% + 15m price below 20 EMA + RSI < 40 + "
        "volume > 2x 20-period average."
    )

    st.markdown("#### Swing rules")
    st.write(
        "Strongest 1W/1M momentum + price above 50/200 EMA + RSI 50-65 + "
        "Debt/Equity < 1.5 + ROE > 15%."
    )

else:
    st.subheader(
        f"{mode} — {len(results)} qualifying stock(s)"
    )

    # Main table exactly around requested fields.
    display_cols = [
        "Ticker",
        "Signal",
        "Current Price",
        "Intraday/Weekly % Change",
        "Reason for Selection",
        "Win Probability (%)",
        "Core Metrics",
    ]

    display_df = results[display_cols].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Signal": st.column_config.TextColumn("Signal"),
            "Current Price": st.column_config.NumberColumn(
                "Current Price", format="₹%.2f"
            ),
            "Intraday/Weekly % Change": st.column_config.NumberColumn(
                "Intraday/Weekly % Change", format="%.2f%%"
            ),
            "Reason for Selection": st.column_config.TextColumn(
                "Reason for Selection", width="large"
            ),
            "Win Probability (%)": st.column_config.NumberColumn(
                "Win Probability (%)", format="%.0f%%"
            ),
            "Core Metrics": st.column_config.TextColumn(
                "Core Metrics", width="large"
            ),
        },
    )

    st.markdown("### Trade Levels")

    levels = results[
        [
            "Ticker",
            "Signal",
            "Current Price",
            "Win Probability (%)",
            "Entry",
            "Stop Loss",
            "Target 1",
            "Target 2",
            "Target 3",
        ]
    ].copy()

    st.dataframe(
        levels,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Win Probability (%)": st.column_config.NumberColumn(format="%.0f%%"),
            "Entry": st.column_config.NumberColumn(format="₹%.2f"),
            "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
            "Target 3": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )

    # Download results.
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Scan Results CSV",
        data=csv,
        file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Important: Yahoo Finance data is not an exchange-grade trading feed and "
    "can be delayed. Intraday signals should be independently verified before "
    "placing any order. No system can guarantee profit."
)
