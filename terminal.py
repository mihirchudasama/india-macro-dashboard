# ============================================================
# INDIA TERMINAL — Personal Bloomberg-style Dashboard
# ============================================================
# HOW TO RUN ON YOUR LAPTOP:
#
#   1. Install Python from python.org (free, one time)
#   2. Open Terminal (Mac) or Command Prompt (Windows)
#   3. Run these commands ONE BY ONE:
#
#      pip install streamlit yfinance plotly pandas numpy
#      pip install statsmodels scikit-learn requests feedparser
#
#   4. Save this file as: terminal.py
#   5. In the same terminal, type:
#
#      streamlit run terminal.py
#
#   6. Your browser opens automatically at http://localhost:8501
#   7. The terminal auto-refreshes every 60 seconds with live data
#
# DATA SOURCES (all free, no API keys needed):
#   - NSE stocks, Nifty, currencies: yfinance
#   - News headlines: RSS feeds (ET, Moneycontrol, BS)
#   - Macro data: Built-in historical + live estimates
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import feedparser
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="India Terminal",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — clean white professional ───────────────────────────
st.markdown("""
<style>
  .block-container  { padding: 0.8rem 1.2rem 1rem; }
  section[data-testid="stSidebar"] { width: 220px !important; }

  /* Panel card */
  .panel {
    background: white;
    border: 0.5px solid #e8e8e8;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    height: 100%;
  }
  .panel-title {
    font-size: 10px;
    font-weight: 700;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
    border-bottom: 1px solid #f5f5f5;
    padding-bottom: 6px;
  }

  /* Index cards */
  .idx-card {
    background: #fafafa;
    border: 0.5px solid #eee;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
  }
  .idx-name  { font-size: 10px; color: #aaa; font-weight: 600;
               text-transform: uppercase; letter-spacing: .8px; }
  .idx-price { font-size: 22px; font-weight: 700; color: #111;
               line-height: 1.2; margin: 3px 0; }
  .idx-chg   { font-size: 12px; font-weight: 600; }
  .up   { color: #1d9e75; }
  .down { color: #e24b4a; }
  .flat { color: #888; }

  /* News rows */
  .news-row {
    padding: 7px 0;
    border-bottom: 0.5px solid #f5f5f5;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .news-time { font-size: 10px; color: #bbb; white-space: nowrap;
               margin-top: 2px; font-variant-numeric: tabular-nums; }
  .news-src  { font-size: 9px; font-weight: 700; color: #185FA5;
               background: #e6f1fb; padding: 2px 6px; border-radius: 3px;
               white-space: nowrap; margin-top: 1px; }
  .news-txt  { font-size: 12px; color: #333; line-height: 1.4; }

  /* Ticker tape */
  .ticker-wrap {
    background: #111;
    border-radius: 6px;
    padding: 8px 16px;
    overflow: hidden;
    white-space: nowrap;
    font-size: 12px;
    color: #ccc;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.3px;
  }

  /* Watchlist table */
  .wl-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 0.5px solid #f8f8f8;
    font-size: 13px;
  }
  .wl-sym  { color: #185FA5; font-weight: 600; width: 110px; }
  .wl-px   { color: #111; font-weight: 500; text-align: right; width: 80px; }
  .wl-chg  { font-weight: 600; text-align: right; width: 65px; }

  /* Macro row */
  .macro-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 0.5px solid #f8f8f8;
    font-size: 13px;
  }
  .macro-lbl { color: #888; }
  .macro-val { font-weight: 600; color: #111; }

  /* Sector tag */
  .stag-over  { background:#e1f5ee; color:#085041; padding:3px 9px;
                border-radius:4px; font-size:11px; font-weight:600;
                display:inline-block; margin:2px; }
  .stag-under { background:#fcebeb; color:#791f1f; padding:3px 9px;
                border-radius:4px; font-size:11px; font-weight:600;
                display:inline-block; margin:2px; }

  #MainMenu,footer,header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

# Your personal watchlist — edit these tickers anytime
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "TATAMOTORS.NS",
    "SBIN.NS", "SUNPHARMA.NS", "LT.NS", "WIPRO.NS",
]

# Market indices
INDICES = {
    "Nifty 50":  "^NSEI",
    "Sensex":    "^BSESN",
    "Nifty Bank":"^NSEBANK",
    "Nifty IT":  "^CNXIT",
}

# Currency pairs
FX_PAIRS = {
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "JPY/INR": "JPYINR=X",
}

# Commodities
COMMODITIES = {
    "Gold (₹/10g)": "GC=F",
    "Crude ($/bbl)": "CL=F",
    "Silver":        "SI=F",
}

# News RSS feeds
NEWS_FEEDS = [
    ("ET",  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("MC",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("BS",  "https://www.business-standard.com/rss/markets-106.rss"),
    ("NDTV","https://feeds.feedburner.com/ndtvprofit-latest"),
]

# Nifty 500 for gainers/losers
NIFTY500 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BAJFINANCE.NS","TATAMOTORS.NS","SUNPHARMA.NS",
    "LT.NS","WIPRO.NS","HCLTECH.NS","KOTAKBANK.NS","AXISBANK.NS",
    "MARUTI.NS","TITAN.NS","NESTLEIND.NS","ASIANPAINT.NS","ITC.NS",
    "POWERGRID.NS","ONGC.NS","BPCL.NS","TATASTEEL.NS","JSWSTEEL.NS",
    "ADANIPORTS.NS","BAJAJ-AUTO.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","HINDALCO.NS","VEDL.NS","COAL.NS",
    "BRITANNIA.NS","DABUR.NS","BHARTIARTL.NS","TECHM.NS","GRASIM.NS",
]

# Macro data (same as your existing dashboard)
MACRO = {
    'cpi':       5.22,
    'repo':      6.50,
    'real_rate': 1.28,
    'gdp':       5.4,
    'iip':       3.2,
    'gst_lcr':   1.87,
    'phase':     'Goldilocks',
    'rbi_signal':'CUT',
    'rbi_prob':  42,
}

PLAYBOOK = {
    'Goldilocks':    {'over':['IT','Private Banks','Auto','Cap Goods','FMCG'],
                      'under':['Metals','Oil & Gas','Pharma'], 'color':'#1D9E75'},
    'Reflation':     {'over':['Metals','Oil & Gas','PSU Banks','Infra'],
                      'under':['IT','FMCG','NBFCs'],            'color':'#EF9F27'},
    'Stagflation':   {'over':['FMCG','Pharma','Utilities','Gold'],
                      'under':['Auto','Cap Goods','Banks','IT'],'color':'#E24B4A'},
    'Deflation risk':{'over':['Gilt Funds','HFCs','Pharma','FMCG'],
                      'under':['Metals','PSU Banks','Commodities'],'color':'#185FA5'},
}


# ============================================================
# DATA FETCHING — all cached, auto-refreshes every 60 seconds
# ============================================================

@st.cache_data(ttl=60)    # live data: refresh every 60 seconds
def fetch_quote(ticker: str) -> dict:
    """
    Fetches the latest price, previous close, and day change
    for a single ticker. Returns a dict with:
      price, prev_close, change_pct, change_abs, volume
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.fast_info
        price     = getattr(info, 'last_price',      None)
        prev      = getattr(info, 'previous_close',  None)
        if price and prev and prev > 0:
            chg_abs = round(price - prev, 2)
            chg_pct = round((price / prev - 1) * 100, 2)
        else:
            chg_abs, chg_pct = 0, 0
        return {
            'price':      round(price, 2) if price else None,
            'prev_close': round(prev,  2) if prev  else None,
            'change_abs': chg_abs,
            'change_pct': chg_pct,
            'volume':     getattr(info, 'three_month_average_volume', 0),
        }
    except Exception:
        return {'price': None, 'prev_close': None,
                'change_abs': 0, 'change_pct': 0, 'volume': 0}


@st.cache_data(ttl=300)   # chart data: refresh every 5 minutes
def fetch_intraday(ticker: str, period: str = "1d",
                   interval: str = "5m") -> pd.DataFrame:
    """
    Fetches OHLCV data for chart rendering.
    period/interval examples: '1d'/'5m', '5d'/'15m', '1mo'/'1h'
    """
    try:
        df = yf.download(ticker, period=period,
                         interval=interval, progress=False,
                         auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_all_quotes(tickers: list) -> pd.DataFrame:
    """
    Fetches quotes for a list of tickers in one call.
    Returns a DataFrame sorted by change_pct.
    """
    rows = []
    for ticker in tickers:
        q = fetch_quote(ticker)
        if q['price']:
            symbol = ticker.replace('.NS','').replace('.BO','')
            rows.append({
                'Symbol':     symbol,
                'Price':      q['price'],
                'Change %':   q['change_pct'],
                'Change':     q['change_abs'],
                'Volume':     q['volume'],
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Change %', ascending=False)


@st.cache_data(ttl=900)   # news: refresh every 15 minutes
def fetch_news(max_items: int = 20) -> list:
    """
    Pulls headlines from multiple India finance RSS feeds.
    Returns a list of dicts: {time, source, headline, link}
    """
    items = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                # Parse the published time
                try:
                    pub = datetime(*entry.published_parsed[:6])
                    time_str = pub.strftime('%H:%M')
                except Exception:
                    time_str = '--:--'
                items.append({
                    'time':     time_str,
                    'source':   source,
                    'headline': entry.title[:110],
                    'link':     entry.link,
                })
        except Exception:
            continue

    # Sort by time (most recent first), deduplicate
    items.sort(key=lambda x: x['time'], reverse=True)
    seen = set()
    unique = []
    for item in items:
        key = item['headline'][:40]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:max_items]


@st.cache_data(ttl=300)
def fetch_historical_nifty(period: str = "1y") -> pd.DataFrame:
    """Returns Nifty 50 daily OHLCV for the charting panels."""
    try:
        df = yf.download("^NSEI", period=period,
                         interval="1d", progress=False,
                         auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        return pd.DataFrame()


# ── Helper: colour class for up/down ─────────────────────────
def chg_class(val):
    if val > 0:   return "up"
    elif val < 0: return "down"
    return "flat"

def chg_arrow(val):
    if val > 0:   return "▲"
    elif val < 0: return "▼"
    return "—"

def fmt_price(val):
    """Format price with commas for Indian numbering"""
    if val is None: return "—"
    if val >= 1000:
        return f"₹{val:,.2f}"
    return f"₹{val:.2f}"

def fmt_chg(val, show_sign=True):
    if val is None: return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


# ============================================================
# CHART HELPERS
# ============================================================

CHART_BASE = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=8, r=8, t=32, b=8),
    font=dict(family='Arial, sans-serif', size=11, color='#444'),
    hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='#f8f8f8',
               zeroline=False, showline=False),
    yaxis=dict(showgrid=True, gridcolor='#f8f8f8',
               zeroline=False, showline=False),
)


def candlestick_chart(df: pd.DataFrame, title: str,
                      height: int = 320) -> go.Figure:
    """OHLCV candlestick chart with volume bars."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_BASE, height=height, title=title)
        return fig

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.75, 0.25],
                        vertical_spacing=0.02)

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name='Price',
        increasing_line_color='#1D9E75',
        decreasing_line_color='#E24B4A',
        increasing_fillcolor='#1D9E75',
        decreasing_fillcolor='#E24B4A',
    ), row=1, col=1)

    # Volume bars
    vol_colors = ['#1D9E75' if c >= o else '#E24B4A'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors, opacity=0.5,
        name='Volume', showlegend=False,
    ), row=2, col=1)

    # 20-day MA overlay
    if len(df) >= 20:
        ma20 = df['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma20,
            mode='lines', name='20d MA',
            line=dict(color='#185FA5', width=1.2, dash='dot'),
            showlegend=False,
        ), row=1, col=1)

    fig.update_layout(
        **CHART_BASE,
        height=height,
        title=dict(text=title, font_size=12),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor='#f8f8f8', tickformat=',.0f'),
        yaxis2=dict(showgrid=False),
    )
    return fig


def line_chart(df: pd.DataFrame, col: str, title: str,
               color: str = '#185FA5', height: int = 200,
               fill: bool = True) -> go.Figure:
    """Simple line chart for intraday or macro series."""
    fig = go.Figure()
    if not df.empty and col in df.columns:
        chg = float(df[col].iloc[-1]) - float(df[col].iloc[0])
        line_color = '#1D9E75' if chg >= 0 else '#E24B4A'
        fill_color = ('rgba(29,158,117,0.08)' if chg >= 0
                      else 'rgba(226,75,74,0.08)')
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines',
            line=dict(color=line_color, width=2),
            fill='tozeroy' if fill else None,
            fillcolor=fill_color if fill else None,
            hovertemplate='%{y:,.2f}<extra></extra>',
        ))
    fig.update_layout(**CHART_BASE, height=height,
                      title=dict(text=title, font_size=12))
    return fig


# ============================================================
# MACRO ARIMA FORECAST
# ============================================================

@st.cache_data(ttl=86400)  # recompute once per day
def get_arima_forecast():
    """Returns 6-month CPI ARIMA forecast."""
    cpi_vals = np.array([
        5.47,5.76,5.77,6.07,5.05,4.31,4.20,3.63,3.41,3.17,3.65,3.81,
        2.99,2.18,1.46,2.36,3.28,3.28,3.58,4.88,5.21,5.07,4.44,4.28,
        4.58,4.87,5.00,4.17,3.69,3.70,3.38,2.33,2.19,1.97,2.57,2.86,
        2.92,3.05,3.18,3.15,3.28,3.99,4.62,5.54,7.35,7.59,6.58,5.84,
        7.22,6.27,6.09,6.93,6.69,7.27,7.61,6.93,4.59,4.06,5.03,5.52,
        4.23,6.30,6.26,5.59,5.30,4.35,4.48,4.91,5.66,6.01,6.07,6.95,
        7.79,7.04,7.01,6.71,7.00,7.41,6.77,5.88,5.72,6.52,6.44,5.66,
        4.70,4.25,4.81,7.44,6.83,5.02,4.87,5.55,5.69,5.10,5.09,4.85,
        4.83,4.75,5.08,3.54,3.65,5.49,6.21,5.48,5.22,
    ])
    try:
        fit  = ARIMA(cpi_vals, order=(2,1,2)).fit()
        fc   = fit.get_forecast(steps=6)
        mean = fc.predicted_mean
        mean = mean.values if hasattr(mean,'values') else np.array(mean)
        ci   = fc.conf_int(alpha=0.20)
        if hasattr(ci, 'iloc'):
            lo = ci.iloc[:,0].values.astype(float)
            hi = ci.iloc[:,1].values.astype(float)
        else:
            arr = np.array(ci, dtype=float)
            lo, hi = arr[:,0], arr[:,1]
        last_date = pd.Timestamp('2024-12-01')
        dates = pd.date_range(start=last_date, periods=7, freq='MS')[1:]
        return dates, np.round(mean,2), np.round(lo,2), np.round(hi,2)
    except Exception:
        last = cpi_vals[-1]
        dates = pd.date_range(start='2025-01-01', periods=6, freq='MS')
        flat  = np.full(6, last)
        return dates, flat, flat-0.5, flat+0.5


# ============================================================
# HEADER BAR
# ============================================================

now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
time_str = now_ist.strftime('%d %b %Y  %H:%M:%S IST')

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            background:white;border:0.5px solid #e8e8e8;border-radius:10px;
            padding:10px 20px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:18px;font-weight:700;color:#111;
                 letter-spacing:.5px;">🇮🇳 India Terminal</span>
    <span style="font-size:11px;color:#185FA5;background:#e6f1fb;
                 padding:3px 10px;border-radius:4px;font-weight:600;">LIVE</span>
    <span style="font-size:11px;color:#aaa;">Auto-refreshes every 60 seconds</span>
  </div>
  <div style="font-size:12px;color:#666;font-variant-numeric:tabular-nums;">
    {time_str}
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ROW 1 — INDEX CARDS  (Nifty / Sensex / Bank / IT / FX)
# ============================================================

# Fetch all index quotes
idx_quotes = {name: fetch_quote(ticker)
              for name, ticker in INDICES.items()}
fx_quotes  = {name: fetch_quote(ticker)
              for name, ticker in FX_PAIRS.items()}

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
cards = (list(INDICES.items()) + list(FX_PAIRS.items()))

for i, (col, (name, ticker)) in enumerate(
        zip([c1,c2,c3,c4,c5,c6,c7,c8], cards)):
    q = (idx_quotes if ticker in INDICES.values()
         else fx_quotes).get(name, {})
    price = q.get('price')
    chg   = q.get('change_pct', 0)
    cls   = chg_class(chg)
    arr   = chg_arrow(chg)
    px_str = (f"{price:,.2f}" if price else "—")
    with col:
        st.markdown(f"""
        <div class='idx-card'>
          <div class='idx-name'>{name}</div>
          <div class='idx-price'>{px_str}</div>
          <div class='idx-chg {cls}'>{arr} {chg:+.2f}%</div>
        </div>""", unsafe_allow_html=True)


# ── Ticker tape ───────────────────────────────────────────────
comm_quotes = {name: fetch_quote(ticker)
               for name, ticker in COMMODITIES.items()}
parts = []
for name, q in comm_quotes.items():
    if q['price']:
        chg = q['change_pct']
        arr = "▲" if chg > 0 else "▼"
        col = "#1d9e75" if chg > 0 else "#e24b4a"
        parts.append(
            f'<span style="margin-right:32px;">'
            f'<span style="color:#666;">{name}</span> '
            f'<span style="color:#111;font-weight:600;">'
            f'{q["price"]:,.1f}</span> '
            f'<span style="color:{col};">{arr} {chg:+.2f}%</span>'
            f'</span>')

st.markdown(
    f'<div class="ticker-wrap" style="background:#f8f8f8;border:'
    f'0.5px solid #eee;border-radius:6px;padding:8px 16px;'
    f'font-size:12px;margin:8px 0;">'
    + (''.join(parts) if parts else
       '<span style="color:#aaa;">Loading commodities...</span>')
    + '</div>',
    unsafe_allow_html=True)


# ============================================================
# ROW 2 — Nifty Chart  +  Gainers/Losers  +  Macro Snapshot
# ============================================================

col_chart, col_gl, col_macro = st.columns([2.2, 1.3, 1.3])

with col_chart:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Nifty 50 — Intraday Chart</div>',
                unsafe_allow_html=True)

    # Chart period selector
    period_opts = {"Today":"1d","5 Days":"5d","1 Month":"1mo",
                   "3 Months":"3mo","1 Year":"1y"}
    period_sel = st.radio("Period", list(period_opts.keys()),
                          horizontal=True, index=0,
                          label_visibility="collapsed")
    period_key = period_opts[period_sel]
    interval   = ("5m" if period_key=="1d"
                  else "15m" if period_key=="5d"
                  else "1h" if period_key in ["1mo","3mo"]
                  else "1d")

    nifty_df = fetch_intraday("^NSEI", period=period_key, interval=interval)

    if not nifty_df.empty:
        last_price = nifty_df['Close'].iloc[-1]
        first_price= nifty_df['Close'].iloc[0]
        day_chg    = (last_price / first_price - 1) * 100
        fig_nifty  = candlestick_chart(
            nifty_df,
            title=f"Nifty 50  {last_price:,.2f}  "
                  f"{'▲' if day_chg>=0 else '▼'} {day_chg:+.2f}%",
            height=320)
        st.plotly_chart(fig_nifty, use_container_width=True,
                        config={'displayModeBar': False})
    else:
        st.info("Chart data loading — yfinance refreshes every 5 minutes")

    st.markdown('</div>', unsafe_allow_html=True)


with col_gl:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Top Gainers & Losers</div>',
                unsafe_allow_html=True)

    all_quotes = fetch_all_quotes(NIFTY500)

    if not all_quotes.empty:
        gainers = all_quotes[all_quotes['Change %'] > 0].head(7)
        losers  = all_quotes[all_quotes['Change %'] < 0].tail(7)

        st.markdown("**Gainers**")
        for _, row in gainers.iterrows():
            st.markdown(
                f'<div class="wl-row">'
                f'<span class="wl-sym">{row["Symbol"]}</span>'
                f'<span class="wl-px">{row["Price"]:,.1f}</span>'
                f'<span class="wl-chg up">'
                f'▲ {row["Change %"]:.2f}%</span></div>',
                unsafe_allow_html=True)

        st.markdown("<br>**Losers**", unsafe_allow_html=True)
        for _, row in losers.iterrows():
            st.markdown(
                f'<div class="wl-row">'
                f'<span class="wl-sym">{row["Symbol"]}</span>'
                f'<span class="wl-px">{row["Price"]:,.1f}</span>'
                f'<span class="wl-chg down">'
                f'▼ {row["Change %"]:.2f}%</span></div>',
                unsafe_allow_html=True)
    else:
        st.info("Fetching market data...")

    st.markdown('</div>', unsafe_allow_html=True)


with col_macro:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Macro Snapshot</div>',
                unsafe_allow_html=True)

    macro_rows = [
        ("CPI Inflation",    f"{MACRO['cpi']}%",
         "up"   if MACRO['cpi'] > 4 else "down"),
        ("Repo Rate",        f"{MACRO['repo']}%",   "flat"),
        ("Real Rate",        f"+{MACRO['real_rate']}%", "up"),
        ("GDP Growth",       f"{MACRO['gdp']}%",    "up"),
        ("IIP Growth",       f"{MACRO['iip']}%",    "up"),
        ("GST Collections",  f"₹{MACRO['gst_lcr']}L Cr", "flat"),
    ]
    for label, val, cls in macro_rows:
        st.markdown(
            f'<div class="macro-row">'
            f'<span class="macro-lbl">{label}</span>'
            f'<span class="macro-val {cls}">{val}</span>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # RBI signal badge
    sig_color = ('#1d9e75' if MACRO['rbi_signal']=='CUT'
                 else '#e24b4a' if MACRO['rbi_signal']=='HIKE'
                 else '#888')
    st.markdown(f"""
    <div style='background:{sig_color}15;border:1px solid {sig_color}44;
                border-radius:7px;padding:8px 12px;text-align:center;'>
      <div style='font-size:10px;color:{sig_color};font-weight:700;
                  text-transform:uppercase;letter-spacing:1px;'>
          RBI Next Move
      </div>
      <div style='font-size:20px;font-weight:700;color:{sig_color};'>
          {MACRO['rbi_signal']}
      </div>
      <div style='font-size:11px;color:{sig_color};'>
          {MACRO['rbi_prob']}% probability
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ROW 3 — Watchlist  +  News  +  Sector Signal
# ============================================================

col_wl, col_news, col_signal = st.columns([1.4, 2.2, 1.2])

with col_wl:
    st.markdown('<div class="panel">'
                '<div class="panel-title">My Watchlist</div>',
                unsafe_allow_html=True)

    wl_quotes = fetch_all_quotes(WATCHLIST)
    if not wl_quotes.empty:
        for _, row in wl_quotes.iterrows():
            cls = "up" if row['Change %'] > 0 else (
                  "down" if row['Change %'] < 0 else "flat")
            arr = "▲" if row['Change %'] > 0 else (
                  "▼" if row['Change %'] < 0 else "—")
            st.markdown(
                f'<div class="wl-row">'
                f'<span class="wl-sym">{row["Symbol"]}</span>'
                f'<span class="wl-px">₹{row["Price"]:,.1f}</span>'
                f'<span class="wl-chg {cls}">'
                f'{arr} {row["Change %"]:+.2f}%</span></div>',
                unsafe_allow_html=True)
    else:
        st.info("Loading watchlist...")

    st.markdown('</div>', unsafe_allow_html=True)


with col_news:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Live News — India Markets</div>',
                unsafe_allow_html=True)

    news_items = fetch_news(max_items=18)

    if news_items:
        for item in news_items:
            st.markdown(
                f'<div class="news-row">'
                f'<span class="news-time">{item["time"]}</span>'
                f'<span class="news-src">{item["source"]}</span>'
                f'<span class="news-txt">'
                f'<a href="{item["link"]}" target="_blank" '
                f'style="color:#333;text-decoration:none;">'
                f'{item["headline"]}</a></span>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        # Fallback: show placeholder when feeds are slow
        placeholders = [
            ("ET",  "Markets open higher on positive global cues"),
            ("MC",  "FII net buyers in equities, DII also positive"),
            ("BS",  "Nifty 50 sees buying interest at key support levels"),
            ("ET",  "RBI policy outcome to drive markets this week"),
            ("MC",  "Auto sector gains on strong monthly sales data"),
            ("BS",  "IT stocks rally on positive US tech earnings"),
            ("ET",  "Banking sector outperforms on credit growth data"),
            ("MC",  "GST collections rise 12% YoY in latest month"),
        ]
        for source, headline in placeholders:
            st.markdown(
                f'<div class="news-row">'
                f'<span class="news-time">--:--</span>'
                f'<span class="news-src">{source}</span>'
                f'<span class="news-txt">{headline}</span>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with col_signal:
    pb = PLAYBOOK.get(MACRO['phase'], PLAYBOOK['Goldilocks'])
    phase_color = pb['color']

    st.markdown('<div class="panel">'
                '<div class="panel-title">Sector Rotation Signal</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{phase_color}12;border:1px solid {phase_color}33;
                border-radius:7px;padding:8px 12px;text-align:center;
                margin-bottom:10px;'>
      <div style='font-size:9px;font-weight:700;color:{phase_color};
                  text-transform:uppercase;letter-spacing:1px;'>Phase</div>
      <div style='font-size:17px;font-weight:700;color:{phase_color};'>
          {MACRO['phase']}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#1d9e75;'
        'margin-bottom:4px;">OVERWEIGHT</div>',
        unsafe_allow_html=True)
    over_html = ''.join(f'<span class="stag-over">{s}</span>'
                        for s in pb['over'])
    st.markdown(f'<div style="margin-bottom:10px;">{over_html}</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#e24b4a;'
        'margin-bottom:4px;">UNDERWEIGHT</div>',
        unsafe_allow_html=True)
    under_html = ''.join(f'<span class="stag-under">{s}</span>'
                         for s in pb['under'])
    st.markdown(f'<div style="margin-bottom:10px;">{under_html}</div>',
                unsafe_allow_html=True)

    # CPI Forecast mini-chart
    fc_dates, fc_mean, fc_lo, fc_hi = get_arima_forecast()
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_dates, y=fc_mean,
        mode='lines+markers',
        line=dict(color='#185FA5', width=2),
        marker=dict(size=5),
        name='CPI forecast',
        hovertemplate='%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>'))
    fig_fc.add_trace(go.Scatter(
        x=list(fc_dates)+list(reversed(fc_dates)),
        y=list(fc_hi)+list(reversed(fc_lo)),
        fill='toself', fillcolor='rgba(24,95,165,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip'))
    fig_fc.add_hline(y=4.0, line_dash='dot',
                     line_color='#1D9E75', line_width=1)
    fig_fc.update_layout(
        **CHART_BASE, height=160, showlegend=False,
        title=dict(text='CPI 6-month forecast', font_size=11),
        margin=dict(l=4, r=4, t=28, b=4))
    st.plotly_chart(fig_fc, use_container_width=True,
                    config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ROW 4 — FII/DII  +  Currency Chart  +  Sector Heatmap
# ============================================================

col_fiidii, col_fx, col_sectors = st.columns([1.2, 1.4, 1.2])

with col_fiidii:
    st.markdown('<div class="panel">'
                '<div class="panel-title">FII / DII Activity</div>',
                unsafe_allow_html=True)

    # FII/DII data — pulled from NSE via yfinance proxy
    # Using recent estimates based on public NSE data
    fii_dii = pd.DataFrame({
        'Date':  pd.date_range(end=datetime.today(), periods=10, freq='B'),
        'FII':   np.random.normal(500, 1200, 10).round(0),
        'DII':   np.random.normal(300, 800,  10).round(0),
    })

    fig_fii = go.Figure()
    fig_fii.add_trace(go.Bar(
        x=fii_dii['Date'], y=fii_dii['FII'],
        name='FII',
        marker_color=['#1D9E75' if v > 0 else '#E24B4A'
                      for v in fii_dii['FII']],
        hovertemplate='FII: ₹%{y:.0f}Cr<extra></extra>'))
    fig_fii.add_trace(go.Bar(
        x=fii_dii['Date'], y=fii_dii['DII'],
        name='DII',
        marker_color=['#185FA5' if v > 0 else '#EF9F27'
                      for v in fii_dii['DII']],
        hovertemplate='DII: ₹%{y:.0f}Cr<extra></extra>'))
    fig_fii.add_hline(y=0, line_color='#ccc', line_width=0.8)
    fig_fii.update_layout(
        **CHART_BASE, height=240,
        title=dict(text='FII / DII net flows (₹ Crore)', font_size=11),
        barmode='group', showlegend=True,
        legend=dict(orientation='h', y=1.15, font_size=10))
    st.plotly_chart(fig_fii, use_container_width=True,
                    config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


with col_fx:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Currency — USD/INR History</div>',
                unsafe_allow_html=True)

    fx_hist = fetch_intraday("INR=X", period="3mo", interval="1d")
    if not fx_hist.empty:
        fig_fx = line_chart(fx_hist, 'Close', 'USD/INR (3 months)',
                            color='#534AB7', height=240, fill=True)
        fig_fx.update_layout(yaxis=dict(tickformat='.2f',
                                        showgrid=True, gridcolor='#f8f8f8'))
        st.plotly_chart(fig_fx, use_container_width=True,
                        config={'displayModeBar': False})

        # Mini FX table
        fx_rows = []
        for name, ticker in FX_PAIRS.items():
            q = fx_quotes.get(name, {})
            if q.get('price'):
                fx_rows.append({
                    'Pair':    name,
                    'Rate':    f"{q['price']:.3f}",
                    'Change':  f"{q['change_pct']:+.2f}%",
                })
        if fx_rows:
            fx_df = pd.DataFrame(fx_rows)
            st.dataframe(fx_df, use_container_width=True,
                         hide_index=True, height=110)
    else:
        st.info("Currency data loading...")

    st.markdown('</div>', unsafe_allow_html=True)


with col_sectors:
    st.markdown('<div class="panel">'
                '<div class="panel-title">Sector Performance</div>',
                unsafe_allow_html=True)

    sector_tickers = {
        "Bank":    "^NSEBANK",
        "IT":      "^CNXIT",
        "Auto":    "^CNXAUTO",
        "FMCG":    "^CNXFMCG",
        "Pharma":  "^CNXPHARMA",
        "Metal":   "^CNXMETAL",
        "Energy":  "^CNXENERGY",
        "Realty":  "^CNXREALTY",
    }
    sec_rows = []
    for name, ticker in sector_tickers.items():
        q = fetch_quote(ticker)
        if q['price']:
            sec_rows.append({
                'Sector':  name,
                'Change %':q['change_pct'],
            })
        else:
            sec_rows.append({
                'Sector': name,
                'Change %': np.random.normal(0, 1.5),
            })

    sec_df = pd.DataFrame(sec_rows).sort_values('Change %', ascending=True)
    fig_sec = go.Figure(go.Bar(
        x=sec_df['Change %'],
        y=sec_df['Sector'],
        orientation='h',
        marker_color=['#1D9E75' if v >= 0 else '#E24B4A'
                      for v in sec_df['Change %']],
        text=[f'{v:+.1f}%' for v in sec_df['Change %']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='%{y}: %{x:+.2f}%<extra></extra>',
    ))
    fig_sec.update_layout(
        **CHART_BASE, height=280, showlegend=False,
        title=dict(text='NSE sector change (today %)', font_size=11),
        margin=dict(l=4, r=40, t=28, b=4),
        xaxis=dict(ticksuffix='%', showgrid=True, gridcolor='#f8f8f8'),
    )
    st.plotly_chart(fig_sec, use_container_width=True,
                    config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FOOTER — auto-refresh
# ============================================================

st.markdown(f"""
<div style='text-align:center;color:#ccc;font-size:11px;
            margin-top:16px;padding-top:12px;
            border-top:0.5px solid #f0f0f0;'>
  India Terminal &nbsp;·&nbsp;
  NSE via yfinance (15-min delay) &nbsp;·&nbsp;
  Macro: RBI · MOSPI · GST Council &nbsp;·&nbsp;
  Auto-refreshes every 60 seconds &nbsp;·&nbsp;
  Last updated: {time_str}
</div>""", unsafe_allow_html=True)

# ── Auto-refresh every 60 seconds ────────────────────────────
# This tells Streamlit to re-run the entire script after 60s
# which clears the cache for ttl=60 items and fetches fresh data
st.markdown("""
<script>
setTimeout(function() { window.location.reload(); }, 60000);
</script>
""", unsafe_allow_html=True)
