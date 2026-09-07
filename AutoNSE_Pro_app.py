import warnings
warnings.filterwarnings('ignore')

import json, math, time, os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from SmartApi import SmartConnect
except Exception:
    SmartConnect = None

st.set_page_config(page_title='AutoNSE AI v4', page_icon='📈', layout='wide')

NSE_LIST_URL = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
ANGEL_MASTER_URL = 'https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json'

# ----------------------------
# Credentials / configuration
# ----------------------------
def secret(name, default=''):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

ANGEL_API_KEY = secret('ANGEL_API_KEY')
ANGEL_CLIENT_CODE = secret('ANGEL_CLIENT_CODE')
ANGEL_PASSWORD = secret('ANGEL_PASSWORD')
ANGEL_TOTP_SECRET = secret('ANGEL_TOTP_SECRET')

ANGEL_READY = bool(ANGEL_API_KEY and ANGEL_CLIENT_CODE and ANGEL_PASSWORD and ANGEL_TOTP_SECRET and SmartConnect)

# ----------------------------
# Indicators
# ----------------------------
def finite(x):
    try: return np.isfinite(float(x))
    except Exception: return False

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    rs = au/ad
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df['Close'].shift(1)
    tr = pd.concat([(df['High']-df['Low']), (df['High']-prev).abs(), (df['Low']-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    d = df.copy()
    for c in ['Open','High','Low','Close','Volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['Close']).copy()
    d['EMA20'] = d.Close.ewm(span=20, adjust=False).mean()
    d['EMA50'] = d.Close.ewm(span=50, adjust=False).mean()
    d['EMA200'] = d.Close.ewm(span=200, adjust=False).mean()
    d['RSI14'] = rsi(d.Close)
    d['ATR14'] = atr(d)
    d['VOL20'] = d.Volume.rolling(20).mean()
    d['VWAP20'] = ((d.Close*d.Volume).rolling(20).sum()/d.Volume.rolling(20).sum()).replace([np.inf,-np.inf],np.nan)
    e12 = d.Close.ewm(span=12, adjust=False).mean(); e26 = d.Close.ewm(span=26, adjust=False).mean()
    d['MACD'] = e12-e26; d['MACDSignal'] = d.MACD.ewm(span=9, adjust=False).mean()
    return d

# ----------------------------
# Universe
# ----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_universe():
    try:
        r = requests.get(NSE_LIST_URL, headers={'User-Agent':'Mozilla/5.0'}, timeout=20)
        r.raise_for_status()
        x = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = 'SYMBOL' if 'SYMBOL' in x.columns else x.columns[0]
        syms = x[col].astype(str).str.upper().str.strip().tolist()
        return [s for s in syms if s and s != 'NAN' and not any(k in s for k in ['TEST','DUMMY'])]
    except Exception:
        return ['RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','LT','AXISBANK','BHARTIARTL','ITC','TATAMOTORS','M&M','MARUTI','TITAN','SUNPHARMA','BEL','HAL','TRENT','NTPC','POWERGRID']

@st.cache_data(ttl=86400, show_spinner=False)
def get_angel_master():
    try:
        r = requests.get(ANGEL_MASTER_URL, timeout=60, headers={'User-Agent':'Mozilla/5.0'})
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty: return pd.DataFrame()
        df = df[(df['exch_seg'].astype(str).str.lower()=='nse') & (df['symbol'].astype(str).str.endswith('-EQ'))].copy()
        df['Symbol'] = df['name'].astype(str).str.upper()
        df['Token'] = df['token'].astype(str)
        df['TradingSymbol'] = df['symbol'].astype(str)
        return df[['Symbol','Token','TradingSymbol']].drop_duplicates('Symbol')
    except Exception:
        return pd.DataFrame()

# ----------------------------
# Data providers
# ----------------------------
def yf_symbol(s): return s + '.NS' if not s.endswith('.NS') else s

def yf_download(symbols, period='1y', interval='1d'):
    if yf is None or not symbols: return {}
    tickers=[yf_symbol(s) for s in symbols]
    try:
        raw=yf.download(tickers=tickers, period=period, interval=interval, group_by='ticker', auto_adjust=False, progress=False, threads=True)
    except Exception:
        return {}
    out={}
    if isinstance(raw.columns, pd.MultiIndex):
        lvl=set(raw.columns.get_level_values(0))
        for t in tickers:
            if t not in lvl: continue
            x=raw[t].copy(); x.columns=[str(c).title() for c in x.columns]
            if 'Close' in x: x=x.dropna(subset=['Close'])
            if len(x): out[t.replace('.NS','')]=x
    else:
        x=raw.copy(); x.columns=[str(c).title() for c in x.columns]
        if 'Close' in x: x=x.dropna(subset=['Close'])
        if len(x): out[tickers[0].replace('.NS','')]=x
    return out

@st.cache_resource(show_spinner=False)
def angel_login():
    if not ANGEL_READY: return None
    try:
        import pyotp
        obj=SmartConnect(api_key=ANGEL_API_KEY)
        s=obj.generateSession(ANGEL_CLIENT_CODE, ANGEL_PASSWORD, pyotp.TOTP(ANGEL_TOTP_SECRET).now())
        if not s or not s.get('status'): return None
        return obj
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def angel_candles(symbol, token, interval='ONE_DAY', days=400):
    obj=angel_login()
    if obj is None: return pd.DataFrame()
    end=datetime.now(); start=end-timedelta(days=days)
    p={'exchange':'NSE','symboltoken':str(token),'interval':interval,'fromdate':start.strftime('%Y-%m-%d %H:%M'),'todate':end.strftime('%Y-%m-%d %H:%M')}
    try:
        res=obj.getCandleData(p)
        if not res or not res.get('status') or not res.get('data'): return pd.DataFrame()
        rows=res['data']; d=pd.DataFrame(rows,columns=['Datetime','Open','High','Low','Close','Volume'])
        d['Datetime']=pd.to_datetime(d['Datetime']); d=d.set_index('Datetime')
        for c in ['Open','High','Low','Close','Volume']: d[c]=pd.to_numeric(d[c],errors='coerce')
        return d.dropna(subset=['Close'])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=15, show_spinner=False)
def angel_full_quote(token):
    obj=angel_login()
    if obj is None or not token: return {}
    try:
        res=obj.getMarketData('FULL', {'NSE':[str(token)]})
        if not res or not res.get('status'): return {}
        fetched=(res.get('data') or {}).get('fetched') or []
        return fetched[0] if fetched else {}
    except Exception:
        return {}

# ----------------------------
# Structure: pivots + support/resistance
# ----------------------------
def pivot_levels(df, left=4, right=4):
    d=df.copy()
    highs=[]; lows=[]
    H=d.High.values; L=d.Low.values
    for i in range(left, len(d)-right):
        if H[i] == np.max(H[i-left:i+right+1]): highs.append(float(H[i]))
        if L[i] == np.min(L[i-left:i+right+1]): lows.append(float(L[i]))
    return lows, highs

def cluster_levels(levels, tolerance=0.012):
    vals=sorted([float(x) for x in levels if finite(x) and x>0])
    if not vals: return []
    clusters=[]
    for v in vals:
        if not clusters or abs(v-np.mean(clusters[-1]))/np.mean(clusters[-1])>tolerance:
            clusters.append([v])
        else: clusters[-1].append(v)
    return [float(np.mean(c)) for c in clusters]

def structure_levels(df):
    d=add_indicators(df)
    price=float(d.Close.iloc[-1])
    lows, highs=pivot_levels(d)
    supports=cluster_levels([x for x in lows if x < price] + [d.EMA20.iloc[-1],d.EMA50.iloc[-1],d.EMA200.iloc[-1]])
    resistances=cluster_levels([x for x in highs if x > price] + [d.EMA20.iloc[-1],d.EMA50.iloc[-1],d.EMA200.iloc[-1]])
    supports=[x for x in supports if x < price*0.995]
    resistances=[x for x in resistances if x > price*1.005]
    supports=sorted(supports, reverse=True)
    resistances=sorted(resistances)
    # Include recent swing extrema if no pivot was found.
    if not supports:
        supports=[float(d.Low.tail(20).min())]
    if not resistances:
        resistances=[float(d.High.tail(20).max())]
    return supports, resistances

# ----------------------------
# Trade setup validator
# ----------------------------
def trade_setup(df):
    d=add_indicators(df)
    if len(d)<80: return {'direction':'WATCH','reason':'Insufficient candles'}
    x=d.iloc[-1]; p=float(x.Close); a=float(x.ATR14) if finite(x.ATR14) else p*0.02
    supports,resistances=structure_levels(d)
    sup=supports[0] if supports else p-a
    rs=[r for r in resistances if r>p]
    res1=rs[0] if rs else p+a*2
    res2=rs[1] if len(rs)>1 else p+a*3
    # Trend scores
    bull=bear=0
    if p>x.EMA20: bull+=2
    else: bear+=2
    if p>x.EMA50: bull+=2
    else: bear+=2
    if p>x.EMA200: bull+=2
    else: bear+=2
    if x.EMA20>x.EMA50: bull+=2
    else: bear+=2
    if x.MACD>x.MACDSignal: bull+=2
    else: bear+=2
    if x.RSI14>55: bull+=2
    if x.RSI14<45: bear+=2
    vx=float(x.Volume/x.VOL20) if finite(x.Volume) and finite(x.VOL20) and x.VOL20>0 else 1
    if vx>1.5:
        if x.Close>x.Open: bull+=2
        elif x.Close<x.Open: bear+=2
    # Breakout / pullback context
    breakout = p > res1*1.002 and vx>=1.3
    near_support = abs(p-sup) <= max(a*1.25, p*0.015)
    near_resistance = abs(res1-p) <= max(a*0.9, p*0.012)
    direction='WATCH'; entry=p; sl=None; t1=None; t2=None; rr=None; reason=[]
    if bull>=bear+4:
        if breakout:
            entry=p; t1=res1 if res1>p*1.01 else res2; t2=res2 if res2>t1*1.01 else p+a*3
            sl=max(sup, p-a*1.2)
            rr=(t1-entry)/(entry-sl) if entry>sl else 0
            reason.append('breakout above resistance with volume')
        elif near_support and not near_resistance:
            entry=p; t1=res1; t2=res2; sl=min(sup, p-a*0.8)
            rr=(t1-entry)/(entry-sl) if entry>sl else 0
            reason.append('bullish trend + support proximity')
        elif res1 > p*1.03:
            entry=p; t1=res1; t2=res2; sl=max(sup,p-a*1.0)
            rr=(t1-entry)/(entry-sl) if entry>sl else 0
            reason.append('bullish trend with room to resistance')
    elif bear>=bull+4:
        support1=sup; ss=[s for s in supports if s<p]
        support2=ss[1] if len(ss)>1 else p-a*3
        breakdown=p<support1*0.998 and vx>=1.3
        if breakdown:
            entry=p; t1=support1; t2=support2; sl=min(resistances[0] if resistances else p+a,p+a*1.2)
            rr=(entry-t1)/(sl-entry) if sl>entry else 0
            reason.append('breakdown below support with volume')
        elif abs(p-support1)<=max(a*0.9,p*0.012):
            entry=p; t1=support1; t2=support2; sl=min(res1,p+a*1.0)
            rr=(entry-t1)/(sl-entry) if sl>entry else 0
            reason.append('bearish trend near support')
    if direction=='WATCH' and rr is None:
        pass
    if bull>=bear+4 and rr is not None:
        direction='BUY' if rr>=1.5 and t1>entry*1.01 else 'WAIT'
        if near_resistance and not breakout: reason.append('resistance is close')
    elif bear>=bull+4 and rr is not None:
        direction='SELL' if rr>=1.5 and t1<entry*0.99 else 'WAIT'
    else:
        direction='WATCH'
    score=max(bull,bear)*10 + min(20, int(max(0,min(2.5,rr or 0))*8))
    score=min(100,score)
    return dict(direction=direction, score=score, bull=bull, bear=bear, price=p, entry=entry, sl=sl, t1=t1, t2=t2, rr=rr,
                support=sup, resistance=res1, resistance2=res2, atr=a, volx=vx, rsi=float(x.RSI14),
                ema20=float(x.EMA20), ema50=float(x.EMA50), ema200=float(x.EMA200), macd=float(x.MACD),
                reason='; '.join(reason) if reason else 'No clean trade structure')

# ----------------------------
# Ranking
# ----------------------------
def analyze_symbol(symbol, daily):
    if daily is None or len(daily)<80: return None
    d=add_indicators(daily)
    s=trade_setup(d)
    s['Symbol']=symbol
    s['Day%']=float(d.Close.iloc[-1]/d.Close.iloc[-2]-1)*100 if len(d)>1 else np.nan
    s['Momentum20%']=float(d.Close.iloc[-1]/d.Close.iloc[-21]-1)*100 if len(d)>21 else np.nan
    return s

@st.cache_data(ttl=900, show_spinner=False)
def broad_scan(symbols, mode):
    # Broad stage: daily trend/momentum. Angel mode uses its historical endpoint with cache;
    # research mode uses Yahoo for rapid screening.
    if mode=='Angel One' and ANGEL_READY:
        master=get_angel_master(); mp=dict(zip(master.Symbol,master.Token))
        rows=[]
        # Keep the broad stage bounded to 1000 and process in rate-limit-safe batches.
        for i,s in enumerate(symbols):
            token=mp.get(s)
            if not token: continue
            d=angel_candles(s,token,'ONE_DAY',400)
            if len(d)<80: continue
            a=add_indicators(d); x=a.iloc[-1]
            rows.append({'Symbol':s,'Price':float(x.Close),'RSI':float(x.RSI14),'Ret20%':float(a.Close.iloc[-1]/a.Close.iloc[-21]-1)*100,'Trend':int((x.Close>x.EMA20)+(x.Close>x.EMA50)+(x.Close>x.EMA200)),'VolX':float(x.Volume/x.VOL20) if finite(x.VOL20) and x.VOL20>0 else 1})
        return pd.DataFrame(rows)
    data=yf_download(symbols,'1y','1d')
    rows=[]
    for s,d in data.items():
        if len(d)<80: continue
        a=add_indicators(d); x=a.iloc[-1]
        rows.append({'Symbol':s,'Price':float(x.Close),'RSI':float(x.RSI14),'Ret20%':float(a.Close.iloc[-1]/a.Close.iloc[-21]-1)*100,'Trend':int((x.Close>x.EMA20)+(x.Close>x.EMA50)+(x.Close>x.EMA200)),'VolX':float(x.Volume/x.VOL20) if finite(x.VOL20) and x.VOL20>0 else 1})
    return pd.DataFrame(rows)

# ----------------------------
# Chart
# ----------------------------
# ----------------------------
# Chart — FIXED
# ----------------------------
def make_chart(symbol, df, setup):
    d = add_indicators(df).tail(180)

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d["Open"],
            high=d["High"],
            low=d["Low"],
            close=d["Close"],
            name="Price"
        )
    )

    # EMAs
    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["EMA20"],
            name="EMA20",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["EMA50"],
            name="EMA50",
            line=dict(width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d.index,
            y=d["EMA200"],
            name="EMA200",
            line=dict(width=1.5)
        )
    )

    # Support / Resistance / Entry / SL / Targets
    levels = [
        ("Support", setup.get("support")),
        ("Resistance", setup.get("resistance")),
        ("Resistance 2", setup.get("resistance2")),
        ("Entry", setup.get("entry")),
        ("Stop Loss", setup.get("sl")),
        ("Target 1", setup.get("t1")),
        ("Target 2", setup.get("t2")),
    ]

    # IMPORTANT:
    # Do NOT use fig.add_hline().
    # Plotly Cloud version can throw an AttributeError/ValueError.
    for name, val in levels:
        if finite(val):
            fig.add_shape(
                type="line",
                x0=d.index[0],
                x1=d.index[-1],
                y0=float(val),
                y1=float(val),
                line=dict(
                    width=1,
                    dash="dot"
                )
            )

            fig.add_annotation(
                x=d.index[-1],
                y=float(val),
                text=f"{name}: {float(val):.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11)
            )

    # Title
    rr = setup.get("rr")

    if finite(rr):
        title = (
            f"{symbol} — {setup.get('direction', 'WATCH')} "
            f"| Score {float(setup.get('score', 0)):.0f} "
            f"| R:R {float(rr):.2f}"
        )
    else:
        title = (
            f"{symbol} — {setup.get('direction', 'WATCH')} "
            f"| Score {float(setup.get('score', 0)):.0f}"
        )

    # Reason box
    reason = str(
        setup.get(
            "reason",
            "No clean setup"
        )
    )

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=(
            f"<b>{setup.get('direction', 'WATCH')}</b>"
            f"<br>{reason}"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",
        borderwidth=1
    )

    fig.update_layout(
        title=title,
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig

# ----------------------------
# UI
# ----------------------------
st.title('📈 AutoNSE AI — Structure + Chart Research')
st.caption('1000-stock screening → directional ranking → real support/resistance → R:R validation → annotated chart. Scores are research heuristics, not guaranteed probabilities.')

with st.sidebar:
    st.header('Scanner Controls')
    universe_n=st.slider('NSE liquid universe',100,1000,1000,50)
    deep_n=st.slider('Deep-analysis candidates',10,100,30,5)
    mode=st.selectbox('Data mode',['Angel One','Research (Yahoo)'],index=0 if ANGEL_READY else 1)
    if ANGEL_READY:
        st.success('Angel One credentials detected')
    else:
        st.warning('Angel One not configured — using Research mode')
    run=st.button('🚀 Run Full Market Scan',use_container_width=True)

if run:
    universe=get_nse_universe()[:universe_n]
    if mode=='Angel One' and ANGEL_READY:
        master=get_angel_master()
        if not master.empty:
            universe=[s for s in universe if s in set(master.Symbol)]
    with st.spinner(f'Screening {len(universe)} stocks…'):
        broad=broad_scan(universe,mode)
    if broad.empty:
        st.error('No market data returned. Check the selected data mode and credentials.')
        st.stop()
    # Momentum + trend shortlist, but do not force bullish direction.
    broad['AbsMomentum']=broad['Ret20%'].abs()
    broad=broad.sort_values(['Trend','AbsMomentum','VolX'],ascending=False)
    shortlist=broad.head(max(deep_n*3,deep_n))['Symbol'].tolist()
    if mode=='Angel One' and ANGEL_READY:
        master=get_angel_master(); mp=dict(zip(master.Symbol,master.Token))
        daily_data={s:angel_candles(s,mp[s],'ONE_DAY',400) for s in shortlist if s in mp}
    else:
        daily_data=yf_download(shortlist,'1y','1d')
    results=[]
    for s in shortlist:
        r=analyze_symbol(s,daily_data.get(s))
        if r: results.append(r)
    res=pd.DataFrame(results)
    if res.empty:
        st.error('Deep analysis returned no valid candidates.')
        st.stop()
    # Rank clean setups first, then WATCH/WAIT by score.
    order={'BUY':0,'SELL':1,'WAIT':2,'WATCH':3}
    res['Order']=res['direction'].map(order).fillna(4)
    res=res.sort_values(['Order','score'],ascending=[True,False]).head(deep_n).reset_index(drop=True)
    st.session_state['scan_results']=res
    st.session_state['scan_data']={s:daily_data.get(s) for s in res.Symbol}

res=st.session_state.get('scan_results')
data_map=st.session_state.get('scan_data',{})
if res is not None and not res.empty:
    buys=int((res['direction']=='BUY').sum()); sells=int((res['direction']=='SELL').sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Strong BUY',buys); c2.metric('Strong SELL',sells); c3.metric('WAIT/WATCH',len(res)-buys-sells); c4.metric('Best Score',f"{res.score.max():.0f}")

    cols=['Symbol','direction','score','bull','bear','price','rsi','volx','support','resistance','resistance2','entry','sl','t1','t2','rr','reason']
    table=res[cols].copy(); table.columns=['Symbol','Direction','Score','Bull','Bear','Price','RSI','VolX','Support','Resistance','Resistance 2','Entry','SL','Target 1','Target 2','R:R','Why']
    st.subheader('🏆 Ranked opportunities')
    st.dataframe(table,hide_index=True,use_container_width=True)

    st.subheader('📊 Chart + full reasoning')
    symbol=st.selectbox('Select stock',res.Symbol.tolist())
    setup=res[res.Symbol==symbol].iloc[0].to_dict()
    df=data_map.get(symbol)
    chart_df=df
    if mode=='Angel One' and ANGEL_READY:
        master=get_angel_master(); mp=dict(zip(master.Symbol,master.Token)); token=mp.get(symbol)
        timeframe=st.selectbox('Chart timeframe',['Daily','15-minute'],index=0)
        if timeframe=='15-minute' and token:
            intr=angel_candles(symbol,token,'FIFTEEN_MINUTE',200)
            if not intr.empty: chart_df=intr
        quote=angel_full_quote(token) if token else {}
        if quote:
            q1,q2,q3,q4=st.columns(4)
            q1.metric('Live LTP',f"₹{float(quote.get('ltp',setup['price'])):.2f}")
            q2.metric('Buy Qty',f"{float(quote.get('totBuyQuan',0)):,.0f}")
            q3.metric('Sell Qty',f"{float(quote.get('totSellQuan',0)):,.0f}")
            bq=float(quote.get('totBuyQuan',0)); sq=float(quote.get('totSellQuan',0));
            q4.metric('Order-flow bias','BUYERS' if bq>sq*1.1 else ('SELLERS' if sq>bq*1.1 else 'BALANCED'))
            depth=quote.get('depth',{}) or {}
            buys=depth.get('buy',[])[:5]; sells=depth.get('sell',[])[:5]
            if buys or sells:
                dc1,dc2=st.columns(2)
                dc1.caption('Best 5 buy depth')
                dc1.dataframe(pd.DataFrame(buys),use_container_width=True,hide_index=True)
                dc2.caption('Best 5 sell depth')
                dc2.dataframe(pd.DataFrame(sells),use_container_width=True,hide_index=True)
    if chart_df is not None and not chart_df.empty:
        st.plotly_chart(make_chart(symbol,chart_df,setup),use_container_width=True)
        a,b,c,d,e=st.columns(5)
        a.metric('Support',f"₹{setup['support']:.2f}")
        b.metric('Resistance',f"₹{setup['resistance']:.2f}")
        c.metric('Entry',f"₹{setup['entry']:.2f}")
        d.metric('SL',f"₹{setup['sl']:.2f}" if finite(setup['sl']) else '—')
        e.metric('R:R',f"{setup['rr']:.2f}" if finite(setup['rr']) else '—')
        st.markdown(f"**Decision:** `{setup['direction']}`<br>**Why:** {setup['reason']}<br>**Technical:** RSI {setup['rsi']:.1f} · Vol {setup['volx']:.2f}x · EMA20 {setup['ema20']:.2f} · EMA50 {setup['ema50']:.2f} · EMA200 {setup['ema200']:.2f}", unsafe_allow_html=True)
        if setup['direction'] in ['BUY','SELL']:
            st.success('Trade structure passed the current rule set. This is a research signal, not a guarantee.')
        else:
            st.info('No clean trade yet. The chart explains what is missing rather than forcing a BUY/SELL.')
else:
    st.info('Run the scan. The app will rank stocks and then show an annotated candlestick chart with support, resistance, entry, SL, targets and the reasons behind the decision.')

st.divider()
st.caption('Data-source note: Angel One mode uses SmartAPI market/historical data when configured. Research mode uses Yahoo Finance. Fundamentals/news are intentionally not fabricated; they should be added as separate source-labelled modules in the next stage.')
