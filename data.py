# ============================================================
# data.py  —  ALL DATA FETCHING FOR THE DASHBOARD
# ============================================================
# This file is imported by the main app.
# It fetches live data from yfinance and public APIs,
# and falls back to clean historical data if any source fails.
# ============================================================

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# MACRO DATA — Historical (always available, no internet needed)
# ============================================================

def get_macro_data():
    """
    Returns the master macro DataFrame with all India indicators.
    This is the cleaned historical dataset from Steps 2 & 3.
    The app derives live estimates on top of this.
    """
    cpi_data = {
        "date": [
            "2016-04","2016-05","2016-06","2016-07","2016-08","2016-09",
            "2016-10","2016-11","2016-12","2017-01","2017-02","2017-03",
            "2017-04","2017-05","2017-06","2017-07","2017-08","2017-09",
            "2017-10","2017-11","2017-12","2018-01","2018-02","2018-03",
            "2018-04","2018-05","2018-06","2018-07","2018-08","2018-09",
            "2018-10","2018-11","2018-12","2019-01","2019-02","2019-03",
            "2019-04","2019-05","2019-06","2019-07","2019-08","2019-09",
            "2019-10","2019-11","2019-12","2020-01","2020-02","2020-03",
            "2020-04","2020-05","2020-06","2020-07","2020-08","2020-09",
            "2020-10","2020-11","2020-12","2021-01","2021-02","2021-03",
            "2021-04","2021-05","2021-06","2021-07","2021-08","2021-09",
            "2021-10","2021-11","2021-12","2022-01","2022-02","2022-03",
            "2022-04","2022-05","2022-06","2022-07","2022-08","2022-09",
            "2022-10","2022-11","2022-12","2023-01","2023-02","2023-03",
            "2023-04","2023-05","2023-06","2023-07","2023-08","2023-09",
            "2023-10","2023-11","2023-12","2024-01","2024-02","2024-03",
            "2024-04","2024-05","2024-06","2024-07","2024-08","2024-09",
            "2024-10","2024-11","2024-12",
        ],
        "cpi_yoy": [
            5.47,5.76,5.77,6.07,5.05,4.31,4.20,3.63,3.41,3.17,3.65,3.81,
            2.99,2.18,1.46,2.36,3.28,3.28,3.58,4.88,5.21,5.07,4.44,4.28,
            4.58,4.87,5.00,4.17,3.69,3.70,3.38,2.33,2.19,1.97,2.57,2.86,
            2.92,3.05,3.18,3.15,3.28,3.99,4.62,5.54,7.35,7.59,6.58,5.84,
            7.22,6.27,6.09,6.93,6.69,7.27,7.61,6.93,4.59,4.06,5.03,5.52,
            4.23,6.30,6.26,5.59,5.30,4.35,4.48,4.91,5.66,6.01,6.07,6.95,
            7.79,7.04,7.01,6.71,7.00,7.41,6.77,5.88,5.72,6.52,6.44,5.66,
            4.70,4.25,4.81,7.44,6.83,5.02,4.87,5.55,5.69,5.10,5.09,4.85,
            4.83,4.75,5.08,3.54,3.65,5.49,6.21,5.48,5.22,
        ],
    }
    repo_data = {
        "date": cpi_data["date"],
        "repo_rate": [
            6.50,6.50,6.50,6.50,6.50,6.50,6.25,6.25,6.25,6.25,6.25,6.25,
            6.25,6.25,6.25,6.25,6.00,6.00,6.00,6.00,6.00,6.00,6.00,6.00,
            6.00,6.00,6.25,6.25,6.50,6.50,6.50,6.50,6.50,6.50,6.25,6.25,
            6.00,6.00,5.75,5.75,5.40,5.15,5.15,5.15,5.15,5.15,5.15,4.40,
            4.40,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,
            4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,4.00,
            4.40,4.90,4.90,5.40,5.40,5.90,5.90,6.25,6.25,6.50,6.50,6.50,
            6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,
            6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,6.50,
        ],
    }
    iip_data = {
        "date": cpi_data["date"],
        "iip_yoy": [
            -0.8,1.2,2.1,0.9,2.5,0.7,1.9,-5.1,1.0,3.0,3.1,3.0,
            3.1,1.7,-0.1,0.9,4.5,3.8,2.2,8.4,7.1,7.4,6.8,4.4,
            3.8,3.2,7.0,6.7,4.8,8.2,8.2,0.5,2.4,1.4,0.1,0.1,
            -0.6,3.4,1.2,-4.3,4.6,0.1,-4.3,2.1,0.1,2.0,5.0,-18.7,
            -57.3,-33.9,-15.8,-10.4,-8.0,-0.7,4.0,2.1,1.0,-0.9,5.7,24.2,
            -13.0,29.3,13.6,12.0,12.0,3.1,3.2,1.4,0.4,2.8,6.0,1.1,
            7.0,19.6,12.7,2.4,0.8,3.1,4.2,7.1,4.3,5.2,5.6,1.1,
            4.2,5.2,3.7,5.7,10.3,6.2,11.7,2.4,3.8,3.8,5.7,4.9,
            5.0,6.2,4.2,4.8,0.1,3.1,3.5,5.2,3.2,
        ],
    }
    gdp_data = {
        "date": [
            "2016-04-01","2016-07-01","2016-10-01","2017-01-01",
            "2017-04-01","2017-07-01","2017-10-01","2018-01-01",
            "2018-04-01","2018-07-01","2018-10-01","2019-01-01",
            "2019-04-01","2019-07-01","2019-10-01","2020-01-01",
            "2020-04-01","2020-07-01","2020-10-01","2021-01-01",
            "2021-04-01","2021-07-01","2021-10-01","2022-01-01",
            "2022-04-01","2022-07-01","2022-10-01","2023-01-01",
            "2023-04-01","2023-07-01","2023-10-01","2024-01-01",
            "2024-04-01","2024-07-01","2024-10-01",
        ],
        "gdp_yoy": [
            7.9,7.5,7.3,6.9,8.4,6.2,5.6,6.2,8.2,7.0,7.1,6.6,
            5.8,5.6,4.4,4.1,3.3,-23.8,-7.4,0.5,1.6,20.3,8.4,5.4,
            4.8,13.5,6.3,4.4,6.1,7.8,8.2,8.6,8.4,6.7,5.4,
        ],
    }
    gst_data = {
        "date": [
            "2020-04","2020-05","2020-06","2020-07","2020-08","2020-09",
            "2020-10","2020-11","2020-12","2021-01","2021-02","2021-03",
            "2021-04","2021-05","2021-06","2021-07","2021-08","2021-09",
            "2021-10","2021-11","2021-12","2022-01","2022-02","2022-03",
            "2022-04","2022-05","2022-06","2022-07","2022-08","2022-09",
            "2022-10","2022-11","2022-12","2023-01","2023-02","2023-03",
            "2023-04","2023-05","2023-06","2023-07","2023-08","2023-09",
            "2023-10","2023-11","2023-12","2024-01","2024-02","2024-03",
            "2024-04","2024-05","2024-06","2024-07","2024-08","2024-09",
            "2024-10","2024-11","2024-12",
        ],
        "gst_collections": [
            32172,62151,90917,87422,86449,95480,105155,104963,115174,
            119847,113143,123902,141384,102709,92849,116393,112020,117010,
            130127,132842,131526,144616,133026,142095,167540,141384,144616,
            148995,143612,147686,151718,145867,149507,155922,149577,160122,
            187035,157090,161497,165105,159069,162712,172003,160657,163659,
            176858,168337,184891,209537,172739,174895,182075,174962,173240,
            168337,175385,185100,
        ],
    }

    # Build DataFrames
    df_cpi  = pd.DataFrame(cpi_data)
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])

    df_repo = pd.DataFrame(repo_data)
    df_repo['date'] = pd.to_datetime(df_repo['date'])

    df_iip  = pd.DataFrame(iip_data)
    df_iip['date'] = pd.to_datetime(df_iip['date'])

    df_gdp  = pd.DataFrame(gdp_data)
    df_gdp['date'] = pd.to_datetime(df_gdp['date'])

    df_gst  = pd.DataFrame(gst_data)
    df_gst['date'] = pd.to_datetime(df_gst['date'])
    df_gst['gst_yoy'] = df_gst['gst_collections'].pct_change(12).mul(100).round(2)

    # Merge
    master = df_cpi.copy()
    for df in [df_repo, df_iip, df_gst[['date','gst_collections','gst_yoy']]]:
        master = pd.merge(master, df, on='date', how='outer')
    master = pd.merge(master, df_gdp, on='date', how='left')
    master['gdp_yoy'] = master['gdp_yoy'].ffill()
    master = master.sort_values('date').reset_index(drop=True)

    # Derived columns
    master['real_rate']   = (master['repo_rate'] - master['cpi_yoy']).round(2)
    master['iip_3m_avg']  = master['iip_yoy'].rolling(3).mean().round(2)
    master['cpi_momentum']= master['cpi_yoy'].diff(3).round(2)

    def classify_cycle(row):
        gdp = row['gdp_yoy'] if pd.notna(row['gdp_yoy']) else 5.5
        cpi = row['cpi_yoy'] if pd.notna(row['cpi_yoy']) else 5.0
        if gdp >= 5.5 and cpi < 5.0:   return 'Goldilocks'
        elif gdp >= 5.5 and cpi >= 5.0: return 'Reflation'
        elif gdp < 5.5 and cpi >= 5.0:  return 'Stagflation'
        else:                            return 'Deflation risk'

    master['macro_phase'] = master.apply(classify_cycle, axis=1)
    return master.dropna(subset=['cpi_yoy','repo_rate']).reset_index(drop=True)


# ============================================================
# LIVE MARKET DATA — via yfinance
# ============================================================

# Full Nifty 500 representative list (50 major stocks, 10 sectors)
NIFTY500_TICKERS = {
    # IT
    "TCS.NS":       ("Tata Consultancy Services", "IT"),
    "INFY.NS":      ("Infosys",                   "IT"),
    "WIPRO.NS":     ("Wipro",                      "IT"),
    "HCLTECH.NS":   ("HCL Technologies",           "IT"),
    "TECHM.NS":     ("Tech Mahindra",              "IT"),
    # Banks
    "HDFCBANK.NS":  ("HDFC Bank",                 "Banks"),
    "ICICIBANK.NS": ("ICICI Bank",                "Banks"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank",        "Banks"),
    "AXISBANK.NS":  ("Axis Bank",                  "Banks"),
    "SBIN.NS":      ("State Bank of India",        "Banks"),
    # FMCG
    "HINDUNILVR.NS":("Hindustan Unilever",         "FMCG"),
    "ITC.NS":       ("ITC",                        "FMCG"),
    "NESTLEIND.NS": ("Nestle India",               "FMCG"),
    "BRITANNIA.NS": ("Britannia",                  "FMCG"),
    "DABUR.NS":     ("Dabur India",                "FMCG"),
    # Auto
    "TATAMOTORS.NS":("Tata Motors",                "Auto"),
    "MARUTI.NS":    ("Maruti Suzuki",              "Auto"),
    "BAJAJ-AUTO.NS":("Bajaj Auto",                 "Auto"),
    "HEROMOTOCO.NS":("Hero MotoCorp",              "Auto"),
    "EICHERMOT.NS": ("Eicher Motors",              "Auto"),
    # Pharma
    "SUNPHARMA.NS": ("Sun Pharma",                 "Pharma"),
    "DRREDDY.NS":   ("Dr Reddy's Labs",            "Pharma"),
    "CIPLA.NS":     ("Cipla",                      "Pharma"),
    "DIVISLAB.NS":  ("Divi's Laboratories",        "Pharma"),
    "APOLLOHOSP.NS":("Apollo Hospitals",           "Pharma"),
    # Energy & Oil
    "RELIANCE.NS":  ("Reliance Industries",        "Energy"),
    "ONGC.NS":      ("ONGC",                       "Energy"),
    "BPCL.NS":      ("BPCL",                       "Energy"),
    "IOC.NS":       ("Indian Oil",                 "Energy"),
    "POWERGRID.NS": ("Power Grid",                 "Energy"),
    # Metals
    "TATASTEEL.NS": ("Tata Steel",                 "Metals"),
    "JSWSTEEL.NS":  ("JSW Steel",                  "Metals"),
    "HINDALCO.NS":  ("Hindalco",                   "Metals"),
    "VEDL.NS":      ("Vedanta",                    "Metals"),
    "COAL.NS":      ("Coal India",                 "Metals"),
    # Capital Goods / Infra
    "LT.NS":        ("Larsen & Toubro",            "Capital Goods"),
    "ADANIPORTS.NS":("Adani Ports",                "Capital Goods"),
    "SIEMENS.NS":   ("Siemens India",              "Capital Goods"),
    "ABB.NS":       ("ABB India",                  "Capital Goods"),
    "BHEL.NS":      ("BHEL",                       "Capital Goods"),
    # NBFC / Finance
    "BAJFINANCE.NS":("Bajaj Finance",              "NBFC"),
    "BAJAJFINSV.NS":("Bajaj Finserv",              "NBFC"),
    "CHOLAFIN.NS":  ("Cholamandalam Finance",      "NBFC"),
    "MUTHOOTFIN.NS":("Muthoot Finance",            "NBFC"),
    "SHRIRAMFIN.NS":("Shriram Finance",            "NBFC"),
    # Telecom / Misc
    "BHARTIARTL.NS":("Bharti Airtel",              "Telecom"),
    "ASIANPAINT.NS":("Asian Paints",               "Consumer"),
    "TITAN.NS":     ("Titan Company",              "Consumer"),
    "ULTRACEMCO.NS":("UltraTech Cement",           "Cement"),
    "GRASIM.NS":    ("Grasim Industries",          "Cement"),
}

SECTOR_INDICES = {
    "Nifty 50":    "^NSEI",
    "Nifty Bank":  "^NSEBANK",
    "Nifty IT":    "^CNXIT",
    "Nifty Auto":  "^CNXAUTO",
    "Nifty FMCG":  "^CNXFMCG",
    "Nifty Pharma":"^CNXPHARMA",
    "Nifty Metal":  "^CNXMETAL",
    "Nifty Energy": "^CNXENERGY",
}


def fetch_stock_data(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Downloads OHLCV data for a list of tickers from yfinance.
    Returns a DataFrame of closing prices.
    On error returns empty DataFrame.
    """
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]]
            closes.columns = tickers
        return closes.dropna(how="all")
    except Exception as e:
        return pd.DataFrame()


def compute_screener(closes: pd.DataFrame,
                     ticker_meta: dict) -> pd.DataFrame:
    """
    Given a DataFrame of closing prices, compute:
      - Momentum:  1m, 3m, 6m, 12m returns
      - Volatility: 30-day rolling std (annualised)
      - RSI:        14-day
      - Composite score: momentum + low-volatility factors

    Returns a sorted screener DataFrame.
    """
    if closes.empty or len(closes) < 30:
        return pd.DataFrame()

    rows = []
    for ticker in closes.columns:
        s = closes[ticker].dropna()
        if len(s) < 30:
            continue

        price = float(s.iloc[-1])

        def ret(days):
            if len(s) > days:
                return round((s.iloc[-1] / s.iloc[-days] - 1) * 100, 2)
            return np.nan

        r1m  = ret(21)
        r3m  = ret(63)
        r6m  = ret(126)
        r12m = ret(252)

        # Annualised volatility
        daily_ret = s.pct_change().dropna()
        vol_30d   = round(daily_ret.tail(30).std() * (252**0.5) * 100, 1)

        # RSI 14
        delta  = s.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = round(float(100 - 100 / (1 + rs.iloc[-1])), 1) if not rs.empty else 50.0

        # Composite score (momentum - volatility)
        mom_score  = np.nanmean([r1m or 0, (r3m or 0)/3, (r6m or 0)/6]) if r3m else 0
        vol_score  = max(0, 50 - vol_30d)
        composite  = round(mom_score * 0.6 + vol_score * 0.4, 1)

        meta = ticker_meta.get(ticker, (ticker, "Unknown"))
        rows.append({
            "Ticker":     ticker,
            "Name":       meta[0],
            "Sector":     meta[1],
            "Price (₹)":  round(price, 1),
            "1M %":       r1m,
            "3M %":       r3m,
            "6M %":       r6m,
            "12M %":      r12m,
            "Volatility": vol_30d,
            "RSI":        rsi,
            "Score":      composite,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df


def fetch_sector_returns(period: str = "1y") -> pd.DataFrame:
    """
    Fetches NSE sector index returns via yfinance.
    Returns a DataFrame with sector names and returns.
    """
    tickers = list(SECTOR_INDICES.values())
    names   = list(SECTOR_INDICES.keys())
    try:
        raw = yf.download(tickers, period=period,
                          interval="1d", progress=False,
                          auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]]
            closes.columns = tickers

        rows = []
        for ticker, name in zip(tickers, names):
            if ticker not in closes.columns:
                continue
            s = closes[ticker].dropna()
            if len(s) < 5:
                continue
            def ret(d):
                if len(s) > d:
                    return round((s.iloc[-1]/s.iloc[-d]-1)*100, 2)
                return np.nan
            rows.append({
                "Sector": name,
                "1M %":   ret(21),
                "3M %":   ret(63),
                "6M %":   ret(126),
                "1Y %":   ret(252),
            })
        return pd.DataFrame(rows) if rows else _sector_fallback()
    except Exception:
        return _sector_fallback()


def _sector_fallback() -> pd.DataFrame:
    """Static fallback sector data when yfinance unavailable."""
    return pd.DataFrame({
        "Sector": ["Nifty 50","Nifty Bank","Nifty IT",
                   "Nifty Auto","Nifty FMCG","Nifty Pharma",
                   "Nifty Metal","Nifty Energy"],
        "1M %":  [1.2, 0.8, 2.1, 1.5, 0.9, 1.8, -0.5, 0.3],
        "3M %":  [4.5, 3.2, 7.8, 5.1, 2.8, 6.2, -1.2, 1.8],
        "6M %":  [8.3, 6.1, 14.2, 9.8, 5.4, 11.3, -3.1, 4.2],
        "1Y %":  [18.5, 12.3, 28.4, 22.1, 10.2, 19.8, -5.4, 8.7],
    })


def fetch_nifty50_history(period: str = "5y") -> pd.DataFrame:
    """Returns Nifty 50 index OHLCV history."""
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return pd.DataFrame()
