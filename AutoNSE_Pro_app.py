import warnings
warnings.filterwarnings('ignore')

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='AutoNSE AI Stock Research', page_icon='📊', layout='wide')

NSE_LIST_URL = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
NIFTY = '^NSEI'
FALLBACK = [
    'RELIANCE','TCS','INFY','HDFCBANK','ICICIBANK','SBIN','TATAMOTORS','AXISBANK','MARUTI','ITC',
    'BHARTIARTL','LT','KOTAKBANK','HINDUNILVR','BAJFINANCE','BAJAJFINSV','SUNPHARMA','TITAN','M&M',
    'HCLTECH','WIPRO','TECHM','TATASTEEL','JSWSTEEL','NTPC','POWERGRID','ONGC','COALINDIA','ADANIENT',
    'ADANIPORTS','ULTRACEMCO','ASIANPAINT','NESTLEIND','TATACONSUM','HINDALCO','GRASIM','CIPLA','DRREDDY',
    'EICHERMOT','HEROMOTOCO','TVSMOTOR','BEL','HAL','TRENT','INDIGO'
]

@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_universe():
    try:
        r = requests.get(NSE_LIST_URL, timeout=15, headers={'User-Agent':'Mozilla/5.0'})
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = 'SYMBOL' if 'SYMBOL' in df.columns else df.columns[0]
        syms = (df[col].astype(str).str.strip().str.upper()
                .replace({'NAN': np.nan}).dropna().tolist())
        syms = [s for s in syms if s.isalnum() or '&' in s or '-' in s]
        if len(syms) >= 800:
            return syms
    except Exception:
        pass
    return FALLBACK

def yf_symbol(s):
    return s if s.endswith('.NS') else s + '.NS'

def clean(s):
    return s.replace('.NS','')

def finite(x):
    try: return np.isfinite(float(x))
    except: return False

def rsi(close, n=14):
    d = close.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev=df['Close'].shift(1)
    tr=pd.concat([(df['High']-df['Low']), (df['High']-prev).abs(), (df['Low']-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def indicators(df):
    d=df.copy()
    if d.empty: return d
    d['EMA20']=d.Close.ewm(span=20,adjust=False).mean()
    d['EMA50']=d.Close.ewm(span=50,adjust=False).mean()
    d['EMA200']=d.Close.ewm(span=200,adjust=False).mean()
    d['RSI14']=rsi(d.Close)
    d['ATR14']=atr(d)
    d['VOL20']=d.Volume.rolling(20).mean()
    d['VWAP']=((d.Close*d.Volume).rolling(20).sum()/d.Volume.rolling(20).sum()).replace([np.inf,-np.inf],np.nan)
    d['RET20']=d.Close.pct_change(20)*100
    d['RET60']=d.Close.pct_change(60)*100
    return d

def download(tickers, period='1y', interval='1d'):
    if not tickers: return {}
    try:
        raw=yf.download(tickers=tickers,period=period,interval=interval,group_by='ticker',auto_adjust=False,progress=False,threads=True)
    except Exception: return {}
    out={}
    if isinstance(raw.columns,pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                x=raw[t].copy(); x.columns=[str(c).title() for c in x.columns]
                x=x.dropna(subset=['Close'])
                if len(x): out[t]=x
    else:
        x=raw.copy(); x.columns=[str(c).title() for c in x.columns]
        if len(x): out[tickers[0]]=x.dropna(subset=['Close'])
    return out

@st.cache_data(ttl=900, show_spinner=False)
def market_snapshot(symbols):
    tickers=[yf_symbol(s) for s in symbols]
    data=download(tickers,'1y','1d')
    rows=[]
    for t,d in data.items():
        d=indicators(d)
        if len(d)<60: continue
        last=d.iloc[-1]; prev=d.iloc[-2]
        price=float(last.Close); day=(price/float(prev.Close)-1)*100
        vol=float(last.Volume) if finite(last.Volume) else np.nan
        av=float(last.VOL20) if finite(last.VOL20) else np.nan
        rows.append({'Symbol':clean(t),'Price':price,'Day%':day,'VolX':vol/av if finite(vol) and finite(av) and av>0 else np.nan,
                     'RSI':float(last.RSI14),'EMA20':float(last.EMA20),'EMA50':float(last.EMA50),'EMA200':float(last.EMA200),
                     'Ret20%':float(last.RET20),'Ret60%':float(last.RET60)})
    return pd.DataFrame(rows)

def technical_score(daily, intraday=None, nifty=None):
    if daily is None or len(daily)<60: return 0, 'Insufficient data'
    d=indicators(daily); x=d.iloc[-1]; p=float(x.Close)
    score=0; reasons=[]
    # Trend
    if p>x.EMA20: score+=8; reasons.append('price above 20 EMA')
    if p>x.EMA50: score+=7; reasons.append('price above 50 EMA')
    if p>x.EMA200: score+=5; reasons.append('price above 200 EMA')
    if x.EMA20>x.EMA50: score+=5; reasons.append('20 EMA above 50 EMA')
    # Momentum
    if 55<=x.RSI14<=70: score+=6; reasons.append(f'RSI {x.RSI14:.0f} supportive')
    elif x.RSI14>70: score+=2; reasons.append(f'RSI {x.RSI14:.0f} strong but stretched')
    elif 45<=x.RSI14<55: score+=3
    if finite(x.RET20) and x.RET20>0: score+=5; reasons.append(f'20D momentum +{x.RET20:.1f}%')
    if finite(x.RET60) and x.RET60>0: score+=5; reasons.append(f'60D momentum +{x.RET60:.1f}%')
    if finite(x.VolX) if False else False: pass
    if intraday is not None and len(intraday)>25:
        q=indicators(intraday); z=q.iloc[-1]
        if z.Close>z.EMA20: score+=6; reasons.append('15m above 20 EMA')
        if z.RSI14>55: score+=4; reasons.append('15m RSI supportive')
        if finite(z.Volume) and finite(z.VOL20) and z.Volume>1.5*z.VOL20: score+=4; reasons.append('15m volume expansion')
    if nifty is not None and len(nifty)>60:
        ni=indicators(nifty); n=ni.iloc[-1]
        if n.Close>n.EMA20 and n.EMA20>n.EMA50: score+=5; reasons.append('NIFTY trend aligned')
    return min(score,55), '; '.join(reasons[:7])

def flow_score(intraday):
    if intraday is None or len(intraday)<25: return 0,'Flow data unavailable'
    d=intraday.copy(); rng=(d.High-d.Low).replace(0,np.nan)
    body=(d.Close-d.Open)/rng
    vol=d.Volume.fillna(0); vavg=vol.rolling(20).mean()
    recent=d.tail(10); rb=body.tail(10); rv=vol.tail(10)
    buy=(rb.clip(lower=0)*rv).sum(); sell=(-rb.clip(upper=0)*rv).sum()
    ratio=buy/(sell+1e-9)
    score=0
    if ratio>1.5: score=15; label='buyers dominant'
    elif ratio>1.15: score=11; label='buyers slightly dominant'
    elif ratio<0.67: score=15; label='sellers dominant'
    elif ratio<0.87: score=11; label='sellers slightly dominant'
    else: score=6; label='balanced flow'
    if finite(vavg.iloc[-1]) and vavg.iloc[-1]>0 and rv.mean()>1.2*vavg.iloc[-1]: score=min(15,score+2)
    return score,label

def fundamental(symbol):
    try:
        info=yf.Ticker(yf_symbol(symbol)).get_info()
        def g(*keys):
            for k in keys:
                if info.get(k) is not None: return info.get(k)
            return np.nan
        roe=g('returnOnEquity'); roa=g('returnOnAssets'); de=g('debtToEquity'); pe=g('trailingPE','forwardPE')
        rev=g('revenueGrowth'); earn=g('earningsGrowth'); margin=g('profitMargins')
        score=0; reasons=[]
        if finite(roe) and roe>0.15: score+=5; reasons.append('ROE >15%')
        if finite(de) and de<150: score+=3; reasons.append('debt/equity acceptable')
        if finite(rev) and rev>0: score+=3; reasons.append('revenue growth positive')
        if finite(earn) and earn>0: score+=3; reasons.append('earnings growth positive')
        if finite(margin) and margin>0.08: score+=2; reasons.append('positive margin')
        return min(score,15), {'ROE':roe*100 if finite(roe) else np.nan,'D/E':de/100 if finite(de) else np.nan,'PE':pe,'RevGrowth':rev*100 if finite(rev) else np.nan,'EarnGrowth':earn*100 if finite(earn) else np.nan}, '; '.join(reasons) or 'Fundamentals unavailable/neutral'
    except Exception:
        return 0, {}, 'Fundamentals unavailable'

def news_score(symbol):
    try:
        news=yf.Ticker(yf_symbol(symbol)).news[:8]
        pos=['beat','growth','order','contract','approval','upgrade','profit','record','expansion','acquisition','buyback','dividend']
        neg=['loss','fraud','downgrade','probe','penalty','default','resign','fall','decline','warning','lawsuit']
        p=n=0; titles=[]
        for item in news:
            title=(item.get('content',{}).get('title') or item.get('title') or '').lower()
            if not title: continue
            titles.append(title[:120])
            p+=sum(w in title for w in pos); n+=sum(w in title for w in neg)
        if p>n and p>0: return 10,'positive news tone',titles[:3]
        if n>p and n>0: return 2,'negative news tone',titles[:3]
        return 6,'mixed/neutral news',titles[:3]
    except Exception: return 5,'news unavailable',[]

def analyze(symbol, daily, intraday, nifty):
    d=indicators(daily); last=d.iloc[-1]; price=float(last.Close)
    tscore, treasons=technical_score(daily,intraday,nifty)
    fscore,fund,freasons=fundamental(symbol)
    nscore,nreason,news=news_score(symbol)
    flow,flowreason=flow_score(intraday)
    market=5 if nifty is not None and len(nifty)>60 and nifty.Close.iloc[-1]>nifty.Close.ewm(span=20,adjust=False).mean().iloc[-1] else 2
    total=tscore+flow+fscore+nscore+market
    direction='BUY' if (last.Close>last.EMA20 and last.RSI14>=50) else 'SELL' if (last.Close<last.EMA20 and last.RSI14<=50) else 'WATCH'
    a=float(last.ATR14) if finite(last.ATR14) else price*0.02
    if direction=='BUY': sl=price-1.2*a; t1=price+1.8*a; t2=price+2.8*a
    elif direction=='SELL': sl=price+1.2*a; t1=price-1.8*a; t2=price-2.8*a
    else: sl=np.nan; t1=np.nan; t2=np.nan
    rr=abs(t1-price)/abs(price-sl) if finite(sl) and finite(t1) else np.nan
    return {'Symbol':symbol,'Direction':direction,'Score':round(total,1),'Price':price,'RSI':float(last.RSI14),'VolX':float(last.Volume/last.VOL20) if finite(last.VOL20) and last.VOL20 else np.nan,'Flow':flowreason,'Technical':treasons,'Fundamental':freasons,'News':nreason,'ROE':fund.get('ROE',np.nan),'D/E':fund.get('D/E',np.nan),'PE':fund.get('PE',np.nan),'Entry':price,'SL':sl,'Target1':t1,'Target2':t2,'RR':rr,'NewsHeadlines':news}

@st.cache_data(ttl=600, show_spinner=False)
def run_scan(max_candidates=40):
    universe=get_nse_universe()
    snap=market_snapshot(tuple(universe))
    if snap.empty: return pd.DataFrame(),0,0
    # Liquidity proxy: select the largest/most active names that Yahoo returned.
    snap=snap.replace([np.inf,-np.inf],np.nan).dropna(subset=['Price','RSI'])
    snap['MomentumRank']=snap['Ret20%'].rank(pct=True).fillna(0)*0.5 + snap['Ret60%'].rank(pct=True).fillna(0)*0.3 + snap['VolX'].rank(pct=True).fillna(0)*0.2
    pre=snap.sort_values('MomentumRank',ascending=False).head(max_candidates)
    syms=pre.Symbol.tolist()
    ddata=download([yf_symbol(s) for s in syms],'1y','1d')
    idata=download([yf_symbol(s) for s in syms],'60d','15m')
    nd=download([NIFTY],'1y','1d'); nifty=nd.get(NIFTY)
    rows=[]
    for s in syms:
        dd=ddata.get(yf_symbol(s)); ii=idata.get(yf_symbol(s))
        if dd is None or ii is None: continue
        try: rows.append(analyze(s,dd,ii,nifty))
        except Exception: continue
    out=pd.DataFrame(rows).sort_values('Score',ascending=False) if rows else pd.DataFrame()
    return out,len(universe),len(pre)

st.title('📊 AutoNSE AI Stock Research & Ranking')
st.caption('Broad NSE screening → deeper technical + flow + fundamentals + news → ranked setups. Scores are research heuristics, not guaranteed probabilities.')

with st.sidebar:
    st.header('Scanner Controls')
    depth=st.slider('Deep-analysis candidates',20,60,40,5)
    run=st.button('🚀 Run Full Market Scan',use_container_width=True)
    st.info('Stage 1 screens the broad NSE equity list. Stage 2 performs deeper analysis on the strongest candidates to keep runtime practical.')

if run:
    with st.spinner('Scanning broad NSE universe and analysing top candidates...'):
        start=time.time(); results,total,pre=run_scan(depth); elapsed=time.time()-start
    st.success(f'Analysed broad universe: {total} stocks → deep analysis: {pre} candidates | {elapsed:.1f}s')
    if results.empty:
        st.warning('No complete dataset was available. Try again later; market-data providers can rate-limit large scans.')
    else:
        buys=results[results.Direction=='BUY'].copy(); sells=results[results.Direction=='SELL'].copy(); watch=results[results.Direction=='WATCH'].copy()
        c1,c2,c3=st.columns(3); c1.metric('Strong BUY candidates',len(buys)); c2.metric('Strong SELL candidates',len(sells)); c3.metric('Watch candidates',len(watch))
        st.subheader('🏆 Top Ranked Opportunities')
        cols=['Symbol','Direction','Score','Price','RSI','VolX','Flow','RR','Entry','SL','Target1','Target2']
        st.dataframe(results[cols].head(15),use_container_width=True,hide_index=True)
        if not buys.empty:
            st.subheader('🟢 Best BUY setups')
            st.dataframe(buys[cols].head(10),use_container_width=True,hide_index=True)
        if not sells.empty:
            st.subheader('🔴 Best SELL setups')
            st.dataframe(sells[cols].head(10),use_container_width=True,hide_index=True)
        st.subheader('🔎 Deep Research')
        pick=st.selectbox('Select a stock',results.Symbol.tolist())
        row=results[results.Symbol==pick].iloc[0]
        a,b,c=st.columns(3); a.metric('Setup Score',row.Score); b.metric('Direction',row.Direction); c.metric('Risk/Reward',f"{row.RR:.2f}" if finite(row.RR) else '-')
        st.markdown(f"**Technical:** {row.Technical}")
        st.markdown(f"**Buyer/Seller flow:** {row.Flow}")
        st.markdown(f"**Fundamentals:** {row.Fundamental}")
        st.markdown(f"**News:** {row.News}")
        st.markdown(f"**Trade framework:** Entry ₹{row.Entry:.2f} | SL ₹{row.SL:.2f} | T1 ₹{row.Target1:.2f} | T2 ₹{row.Target2:.2f}")
        if row.NewsHeadlines: st.write('Recent headlines:', row.NewsHeadlines)
        st.caption('Buyer/seller flow here is an OHLCV-based proxy. True order-book buyer/seller aggression requires broker/exchange feed such as Angel One SmartAPI and will be added in the live-data phase.')
else:
    st.info('Click **Run Full Market Scan** to analyse the broad NSE universe and rank the strongest setups.')
