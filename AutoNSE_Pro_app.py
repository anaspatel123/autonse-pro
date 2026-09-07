
import warnings
warnings.filterwarnings("ignore")

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AutoNSE AI Stock Research", page_icon="📊", layout="wide")

NSE_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NIFTY = "^NSEI"

FALLBACK = [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","TATAMOTORS","AXISBANK","MARUTI","ITC",
    "BHARTIARTL","LT","KOTAKBANK","HINDUNILVR","BAJFINANCE","BAJAJFINSV","SUNPHARMA","TITAN","M&M",
    "HCLTECH","WIPRO","TECHM","TATASTEEL","JSWSTEEL","NTPC","POWERGRID","ONGC","COALINDIA","ADANIENT",
    "ADANIPORTS","ULTRACEMCO","ASIANPAINT","NESTLEIND","TATACONSUM","HINDALCO","GRASIM","CIPLA","DRREDDY",
    "EICHERMOT","HEROMOTOCO","TVSMOTOR","BEL","HAL","TRENT","INDIGO"
]

@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_universe():
    try:
        r = requests.get(
            NSE_LIST_URL, timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        syms = (
            df[col].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan}).dropna().tolist()
        )
        syms = [s for s in syms if s.isalnum() or "&" in s or "-" in s]
        if len(syms) >= 800:
            return syms
    except Exception:
        pass
    return FALLBACK

def yf_symbol(s):
    return s if s.endswith(".NS") else s + ".NS"

def clean(s):
    return s.replace(".NS", "")

def finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    rs = avg_up / avg_dn
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal

def indicators(df):
    d = df.copy()
    if d.empty:
        return d
    d["EMA20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = d.Close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = d.Close.ewm(span=200, adjust=False).mean()
    d["RSI14"] = rsi(d.Close)
    d["ATR14"] = atr(d)
    d["VOL20"] = d.Volume.rolling(20).mean()
    d["VWAP20"] = (
        (d.Close * d.Volume).rolling(20).sum() /
        d.Volume.rolling(20).sum()
    ).replace([np.inf, -np.inf], np.nan)
    d["RET5"] = d.Close.pct_change(5) * 100
    d["RET20"] = d.Close.pct_change(20) * 100
    d["RET60"] = d.Close.pct_change(60) * 100
    d["MACD"], d["MACDSignal"] = macd(d.Close)
    d["AvgTurnover20"] = d.Close * d["VOL20"]
    return d

def download(tickers, period="1y", interval="1d"):
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers, period=period, interval=interval,
            group_by="ticker", auto_adjust=False, progress=False, threads=True
        )
    except Exception:
        return {}

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t not in level0:
                continue
            x = raw[t].copy()
            x.columns = [str(c).title() for c in x.columns]
            if "Close" in x:
                x = x.dropna(subset=["Close"])
            if len(x):
                out[t] = x
    else:
        x = raw.copy()
        x.columns = [str(c).title() for c in x.columns]
        if "Close" in x:
            x = x.dropna(subset=["Close"])
        if len(x):
            out[tickers[0]] = x
    return out

@st.cache_data(ttl=900, show_spinner=False)
def market_snapshot(symbols):
    tickers = [yf_symbol(s) for s in symbols]
    data = download(tickers, "1y", "1d")
    rows = []
    for t, d in data.items():
        d = indicators(d)
        if len(d) < 60:
            continue
        last, prev = d.iloc[-1], d.iloc[-2]
        price = float(last.Close)
        day = (price / float(prev.Close) - 1) * 100
        vol = float(last.Volume) if finite(last.Volume) else np.nan
        av = float(last.VOL20) if finite(last.VOL20) else np.nan
        turnover = float(last.AvgTurnover20) if finite(last.AvgTurnover20) else np.nan
        rows.append({
            "Symbol": clean(t),
            "Price": price,
            "Day%": day,
            "VolX": vol/av if finite(vol) and finite(av) and av > 0 else np.nan,
            "AvgTurnover20": turnover,
            "RSI": float(last.RSI14),
            "EMA20": float(last.EMA20),
            "EMA50": float(last.EMA50),
            "EMA200": float(last.EMA200),
            "Ret5%": float(last.RET5),
            "Ret20%": float(last.RET20),
            "Ret60%": float(last.RET60),
        })
    return pd.DataFrame(rows)

def directional_technical(daily, intraday, nifty):
    """Return separate bullish/bearish technical points out of 40."""
    if daily is None or len(daily) < 60:
        return 0, 0, "Insufficient technical data"

    d = indicators(daily)
    x = d.iloc[-1]
    p = float(x.Close)
    bull = bear = 0
    br, sr = [], []

    # Daily structure: 20 points
    if p > x.EMA20:
        bull += 5; br.append("price > 20 EMA")
    else:
        bear += 5; sr.append("price < 20 EMA")

    if p > x.EMA50:
        bull += 4; br.append("price > 50 EMA")
    else:
        bear += 4; sr.append("price < 50 EMA")

    if p > x.EMA200:
        bull += 3; br.append("price > 200 EMA")
    else:
        bear += 3; sr.append("price < 200 EMA")

    if x.EMA20 > x.EMA50:
        bull += 3; br.append("20 EMA > 50 EMA")
    else:
        bear += 3; sr.append("20 EMA < 50 EMA")

    if x.MACD > x.MACDSignal:
        bull += 2; br.append("MACD bullish")
    else:
        bear += 2; sr.append("MACD bearish")

    if finite(x.RET20):
        if x.RET20 > 2:
            bull += 2; br.append(f"20D +{x.RET20:.1f}%")
        elif x.RET20 < -2:
            bear += 2; sr.append(f"20D {x.RET20:.1f}%")

    if finite(x.RET60):
        if x.RET60 > 5:
            bull += 1; br.append(f"60D +{x.RET60:.1f}%")
        elif x.RET60 < -5:
            bear += 1; sr.append(f"60D {x.RET60:.1f}%")

    # RSI: 8 points, directional rather than bullish-only
    if 52 <= x.RSI14 <= 68:
        bull += 5; br.append(f"RSI {x.RSI14:.0f} bullish zone")
    elif x.RSI14 > 70:
        bull += 2; br.append(f"RSI {x.RSI14:.0f} strong but stretched")
    elif x.RSI14 < 48:
        bear += 5; sr.append(f"RSI {x.RSI14:.0f} weak")
    elif x.RSI14 < 35:
        bear += 2; sr.append(f"RSI {x.RSI14:.0f} oversold")
    else:
        bull += 2; bear += 2

    # 15m confirmation: 12 points
    if intraday is not None and len(intraday) > 25:
        q = indicators(intraday)
        z = q.iloc[-1]
        if z.Close > z.EMA20:
            bull += 4; br.append("15m > 20 EMA")
        else:
            bear += 4; sr.append("15m < 20 EMA")

        if z.EMA20 > z.EMA50:
            bull += 2; br.append("15m EMA structure bullish")
        else:
            bear += 2; sr.append("15m EMA structure bearish")

        if z.RSI14 > 55:
            bull += 2; br.append(f"15m RSI {z.RSI14:.0f}")
        elif z.RSI14 < 45:
            bear += 2; sr.append(f"15m RSI {z.RSI14:.0f}")

        if finite(z.Volume) and finite(z.VOL20) and z.VOL20 > 0:
            vx = z.Volume / z.VOL20
            if vx > 1.5:
                if z.Close >= z.Open:
                    bull += 2; br.append(f"15m volume {vx:.1f}x on up candle")
                else:
                    bear += 2; sr.append(f"15m volume {vx:.1f}x on down candle")

        if z.Close > z.VWAP20:
            bull += 2; br.append("15m above VWAP")
        elif z.Close < z.VWAP20:
            bear += 2; sr.append("15m below VWAP")

    # NIFTY alignment: 5 points
    if nifty is not None and len(nifty) > 60:
        ni = indicators(nifty)
        n = ni.iloc[-1]
        if n.Close > n.EMA20 and n.EMA20 > n.EMA50:
            bull += 5; br.append("NIFTY bullish alignment")
        elif n.Close < n.EMA20 and n.EMA20 < n.EMA50:
            bear += 5; sr.append("NIFTY bearish alignment")
        else:
            bull += 2; bear += 2

    return min(bull, 40), min(bear, 40), (
        "BUY: " + "; ".join(br[:6]) + " | SELL: " + "; ".join(sr[:6])
    )

def directional_flow(intraday):
    """Return BUY and SELL flow points out of 15 using OHLCV proxies."""
    if intraday is None or len(intraday) < 25:
        return 0, 0, "Flow unavailable"

    d = intraday.copy()
    rng = (d.High - d.Low).replace(0, np.nan)
    body = ((d.Close - d.Open) / rng).fillna(0)
    vol = d.Volume.fillna(0)
    recent = d.tail(12)
    rb = body.tail(12)
    rv = vol.tail(12)

    buy = float((rb.clip(lower=0) * rv).sum())
    sell = float((-rb.clip(upper=0) * rv).sum())
    total = buy + sell + 1e-9
    buy_share = buy / total
    sell_share = sell / total

    bull = round(15 * buy_share, 1)
    bear = round(15 * sell_share, 1)

    if buy_share > 0.60:
        label = "buyers dominant"
    elif sell_share > 0.60:
        label = "sellers dominant"
    else:
        label = "balanced flow"

    return bull, bear, label

def fundamental_scores(symbol):
    """Directional fundamental contribution. 10 points total."""
    try:
        info = yf.Ticker(yf_symbol(symbol)).get_info()

        def g(*keys):
            for k in keys:
                if info.get(k) is not None:
                    return info.get(k)
            return np.nan

        roe = g("returnOnEquity")
        de = g("debtToEquity")
        rev = g("revenueGrowth")
        earn = g("earningsGrowth")
        margin = g("profitMargins")
        pe = g("trailingPE", "forwardPE")

        bull = bear = 0
        reasons = []

        if finite(roe):
            if roe > 0.15:
                bull += 2; reasons.append("ROE >15%")
            elif roe < 0:
                bear += 2; reasons.append("negative ROE")

        if finite(de):
            de_ratio = de / 100
            if de_ratio < 1.0:
                bull += 2; reasons.append("low debt")
            elif de_ratio > 2.0:
                bear += 2; reasons.append("high debt")

        if finite(rev):
            if rev > 0:
                bull += 1
            elif rev < 0:
                bear += 1

        if finite(earn):
            if earn > 0:
                bull += 1
            elif earn < 0:
                bear += 1

        if finite(margin):
            if margin > 0.08:
                bull += 1
            elif margin < 0:
                bear += 1

        # Extreme valuation is treated as a risk, not an automatic sell.
        if finite(pe) and pe > 0:
            if pe < 35:
                bull += 1
            elif pe > 80:
                bear += 1

        fund = {
            "ROE": roe * 100 if finite(roe) else np.nan,
            "D/E": de / 100 if finite(de) else np.nan,
            "PE": pe,
            "RevGrowth": rev * 100 if finite(rev) else np.nan,
            "EarnGrowth": earn * 100 if finite(earn) else np.nan,
            "Margin": margin * 100 if finite(margin) else np.nan,
        }
        return min(bull, 10), min(bear, 10), fund, "; ".join(reasons[:5]) or "Fundamentals neutral/unavailable"
    except Exception:
        return 0, 0, {}, "Fundamentals unavailable"

def news_scores(symbol):
    try:
        news = yf.Ticker(yf_symbol(symbol)).news[:10]
        pos = [
            "beat","growth","order","contract","approval","upgrade","profit",
            "record","expansion","acquisition","buyback","dividend","award"
        ]
        neg = [
            "loss","fraud","downgrade","probe","penalty","default","resign",
            "fall","decline","warning","lawsuit","scam","debt","cut"
        ]
        p = n = 0
        titles = []
        for item in news:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = (content.get("title") or item.get("title") or "").strip()
            if not title:
                continue
            low = title.lower()
            titles.append(title[:140])
            p += sum(w in low for w in pos)
            n += sum(w in low for w in neg)

        if p > n and p > 0:
            return 5, 1, "positive news tone", titles[:4]
        if n > p and n > 0:
            return 1, 5, "negative news tone", titles[:4]
        return 3, 3, "mixed/neutral news", titles[:4]
    except Exception:
        return 2, 2, "news unavailable", []

def risk_plan(direction, price, atr_value):
    if not finite(atr_value) or atr_value <= 0:
        atr_value = price * 0.02

    if direction == "BUY":
        sl = price - 1.2 * atr_value
        t1 = price + 1.8 * atr_value
        t2 = price + 2.8 * atr_value
    elif direction == "SELL":
        sl = price + 1.2 * atr_value
        t1 = price - 1.8 * atr_value
        t2 = price - 2.8 * atr_value
    else:
        return np.nan, np.nan, np.nan, np.nan

    rr = abs(t1 - price) / max(abs(price - sl), 1e-9)
    return price, sl, t1, t2, rr

def analyze(symbol, daily, intraday, nifty):
    d = indicators(daily)
    last = d.iloc[-1]
    price = float(last.Close)

    tb, ts, tech_reason = directional_technical(daily, intraday, nifty)
    fb, fs, fund, fund_reason = fundamental_scores(symbol)
    nb, ns, news_reason, headlines = news_scores(symbol)
    flowb, flows, flow_reason = directional_flow(intraday)

    # Small market regime contribution.
    mb = ms = 0
    if nifty is not None and len(nifty) > 60:
        ni = indicators(nifty).iloc[-1]
        if ni.Close > ni.EMA20:
            mb += 5
        if ni.Close < ni.EMA20:
            ms += 5

    buy_score = min(100, tb + flowb + fb + nb + mb)
    sell_score = min(100, ts + flows + fs + ns + ms)

    # Direction requires both a score edge and price structure.
    if buy_score >= 60 and buy_score >= sell_score + 10:
        direction = "BUY"
        score = buy_score
    elif sell_score >= 60 and sell_score >= buy_score + 10:
        direction = "SELL"
        score = sell_score
    else:
        direction = "WATCH"
        score = max(buy_score, sell_score)

    if direction == "BUY":
        entry, sl, t1, t2, rr = risk_plan("BUY", price, float(last.ATR14))
    elif direction == "SELL":
        entry, sl, t1, t2, rr = risk_plan("SELL", price, float(last.ATR14))
    else:
        entry = price
        sl = t1 = t2 = rr = np.nan

    return {
        "Symbol": symbol,
        "Direction": direction,
        "Score": round(float(score), 1),
        "BuyScore": round(float(buy_score), 1),
        "SellScore": round(float(sell_score), 1),
        "Price": price,
        "RSI": round(float(last.RSI14), 1),
        "VolX": round(float(last.Volume / last.VOL20), 2) if finite(last.VOL20) and last.VOL20 > 0 else np.nan,
        "Flow": flow_reason,
        "Technical": tech_reason,
        "Fundamental": fund_reason,
        "News": news_reason,
        "ROE": fund.get("ROE", np.nan),
        "D/E": fund.get("D/E", np.nan),
        "PE": fund.get("PE", np.nan),
        "Entry": entry,
        "SL": sl,
        "Target1": t1,
        "Target2": t2,
        "RR": rr,
        "NewsHeadlines": headlines,
    }

@st.cache_data(ttl=600, show_spinner=False)
def run_scan(top_universe=1000, deep_candidates=60):
    universe = get_nse_universe()
    snap = market_snapshot(tuple(universe))

    if snap.empty:
        return pd.DataFrame(), len(universe), 0, 0

    snap = snap.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Price", "RSI", "AvgTurnover20"]
    )

    # Stage 1: select the most liquid ~1000 names, then rank momentum.
    liquid = snap.sort_values("AvgTurnover20", ascending=False).head(top_universe).copy()

    liquid["MomentumRank"] = (
        liquid["Ret5%"].rank(pct=True).fillna(0) * 0.20 +
        liquid["Ret20%"].rank(pct=True).fillna(0) * 0.35 +
        liquid["Ret60%"].rank(pct=True).fillna(0) * 0.30 +
        liquid["VolX"].rank(pct=True).fillna(0) * 0.15
    )

    pre = liquid.sort_values("MomentumRank", ascending=False).head(deep_candidates)
    syms = pre.Symbol.tolist()

    ddata = download([yf_symbol(s) for s in syms], "1y", "1d")
    idata = download([yf_symbol(s) for s in syms], "60d", "15m")
    nd = download([NIFTY], "1y", "1d")
    nifty = nd.get(NIFTY)

    rows = []
    for s in syms:
        dd = ddata.get(yf_symbol(s))
        ii = idata.get(yf_symbol(s))
        if dd is None:
            continue
        try:
            rows.append(analyze(s, dd, ii, nifty))
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Direction", "Score"], ascending=[True, False])
        # BUY and SELL leaders first, then WATCH.
        order = {"BUY": 0, "SELL": 1, "WATCH": 2}
        out["_o"] = out.Direction.map(order).fillna(9)
        out = out.sort_values(["_o", "Score"], ascending=[True, False]).drop(columns="_o")
    return out, len(universe), len(liquid), len(pre)

st.title("📊 AutoNSE AI Stock Research & Ranking")
st.caption(
    "Broad NSE liquidity screen → momentum shortlist → deep technical + flow + "
    "fundamentals + news → directional ranking. Scores are research heuristics, not guaranteed probabilities."
)

with st.sidebar:
    st.header("Scanner Controls")
    top_universe = st.slider("Liquid NSE universe", 500, 1500, 1000, 100)
    deep = st.slider("Deep-analysis candidates", 20, 100, 60, 10)
    run = st.button("🚀 Run Full Market Scan", use_container_width=True)
    st.info(
        "Stage 1: broad NSE list. Stage 2: most liquid names. "
        "Stage 3: momentum shortlist. Stage 4: deep directional research."
    )

if run:
    with st.spinner("Scanning NSE universe and running deep research..."):
        start = time.time()
        results, total, liquid_count, deep_count = run_scan(top_universe, deep)
        elapsed = time.time() - start

    st.success(
        f"Universe: {total} → liquid: {liquid_count} → deep analysis: "
        f"{deep_count} | {elapsed:.1f}s"
    )

    if results.empty:
        st.warning(
            "No complete dataset was available. Market-data providers can "
            "rate-limit large scans; try again later."
        )
    else:
        buys = results[results.Direction == "BUY"].copy()
        sells = results[results.Direction == "SELL"].copy()
        watch = results[results.Direction == "WATCH"].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Strong BUY", len(buys))
        c2.metric("Strong SELL", len(sells))
        c3.metric("WATCH", len(watch))

        st.subheader("🏆 Top Ranked Opportunities")

        cols = [
            "Symbol","Direction","Score","BuyScore","SellScore","Price","RSI",
            "VolX","Flow","RR","Entry","SL","Target1","Target2"
        ]
        st.dataframe(
            results[cols].head(20),
            use_container_width=True,
            hide_index=True
        )

        if not buys.empty:
            st.subheader("🟢 Best BUY setups")
            st.dataframe(
                buys[cols].head(10),
                use_container_width=True,
                hide_index=True
            )

        if not sells.empty:
            st.subheader("🔴 Best SELL setups")
            st.dataframe(
                sells[cols].head(10),
                use_container_width=True,
                hide_index=True
            )

        if not watch.empty:
            st.subheader("🟡 Watchlist")
            st.dataframe(
                watch[cols].head(10),
                use_container_width=True,
                hide_index=True
            )

        st.subheader("🔎 Deep Research")
        pick = st.selectbox("Select a stock", results.Symbol.tolist())
        row = results[results.Symbol == pick].iloc[0]

        a, b, c, d = st.columns(4)
        a.metric("Setup Score", f"{row.Score:.0f}/100")
        b.metric("BUY score", f"{row.BuyScore:.0f}")
        c.metric("SELL score", f"{row.SellScore:.0f}")
        d.metric("Risk/Reward", f"{row.RR:.2f}" if finite(row.RR) else "-")

        st.markdown(f"**Technical:** {row.Technical}")
        st.markdown(f"**Buyer/Seller flow:** {row.Flow}")
        st.markdown(f"**Fundamentals:** {row.Fundamental}")
        st.markdown(f"**News:** {row.News}")

        if row.NewsHeadlines:
            st.markdown("**Recent headlines:**")
            for h in row.NewsHeadlines:
                st.write("• " + h)

        st.caption(
            "The BUY/SELL flow estimate is an OHLCV proxy, not exchange order-book "
            "data. For true market-depth/order-flow analysis, connect Angel One SmartAPI."
        )
