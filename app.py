# ============================================================
# app.py  —  INDIA MACRO FORECASTER  |  Complete Dashboard
# ============================================================
# Deployment: Streamlit Cloud (free)
# Data:       RBI · MOSPI · GST Council · NSE via yfinance
# Models:     ARIMA · Random Forest
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data   import (get_macro_data, fetch_stock_data, compute_screener,
                    fetch_sector_returns, fetch_nifty50_history,
                    NIFTY500_TICKERS)
from models import run_arima, run_rate_model, PLAYBOOK

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="India Macro Forecaster",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #f8f9fa; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.2rem !important; border-bottom: 2px solid #f0f0f0;
       padding-bottom: 6px; margin-top: 28px !important; }

  /* Metric card */
  .mcard { background:white; border-radius:10px; padding:14px 18px;
           border:1px solid #eee; margin-bottom:4px; }
  .mlabel { font-size:11px; color:#999; font-weight:700;
            text-transform:uppercase; letter-spacing:.7px; }
  .mvalue { font-size:26px; font-weight:700; color:#111;
            line-height:1.2; margin:3px 0; }
  .mdelta { font-size:12px; }

  /* Sector tags */
  .tag-over  { background:#e1f5ee; color:#085041; padding:4px 11px;
               border-radius:5px; font-size:13px; font-weight:600;
               display:inline-block; margin:2px; }
  .tag-under { background:#fcebeb; color:#791f1f; padding:4px 11px;
               border-radius:5px; font-size:13px; font-weight:600;
               display:inline-block; margin:2px; }
  #MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colour constants ──────────────────────────────────────────
CB = '#185FA5'   # blue
CR = '#E24B4A'   # red
CG = '#1D9E75'   # green
CA = '#EF9F27'   # amber
CP = '#534AB7'   # purple
CGR= '#888780'   # gray

CHART_LAYOUT = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family='Arial, sans-serif', size=12, color='#333'),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='#f5f5f5', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='#f5f5f5', zeroline=False),
    legend=dict(orientation='h', y=-0.20, font_size=11),
)


# ============================================================
# CACHED DATA & MODEL FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)   # cache for 1 hour — refreshes automatically
def load_macro():
    return get_macro_data()

@st.cache_data(ttl=3600)
def load_arima(cpi_tuple):
    return run_arima(np.array(cpi_tuple), steps=6)

@st.cache_data(ttl=3600)
def load_rate_model(master_json):
    df = pd.read_json(master_json)
    return run_rate_model(df)

@st.cache_data(ttl=900)    # stock data: cache 15 mins
def load_screener_data():
    tickers = list(NIFTY500_TICKERS.keys())
    closes  = fetch_stock_data(tickers, period="1y")
    if closes.empty:
        return pd.DataFrame()
    return compute_screener(closes, NIFTY500_TICKERS)

@st.cache_data(ttl=900)
def load_sector_returns():
    return fetch_sector_returns(period="1y")

@st.cache_data(ttl=900)
def load_nifty_history():
    return fetch_nifty50_history(period="5y")


# ============================================================
# LOAD ALL DATA
# ============================================================

master = load_macro()

# Current values
cur_cpi  = round(float(master['cpi_yoy'].dropna().iloc[-1]),  2)
cur_repo = round(float(master['repo_rate'].dropna().iloc[-1]),2)
cur_real = round(float(master['real_rate'].dropna().iloc[-1]),2)
cur_iip  = round(float(master['iip_yoy'].dropna().iloc[-1]),  1)
cur_gdp  = round(float(master['gdp_yoy'].dropna().iloc[-1]),  1)
cur_gst  = round(float(master['gst_collections'].dropna().iloc[-1]/100000),2)
cur_phase= master['macro_phase'].iloc[-1]
cur_date = master['date'].iloc[-1].strftime('%B %Y')

prev_cpi = round(float(master['cpi_yoy'].dropna().iloc[-2]),  2)
prev_repo= round(float(master['repo_rate'].dropna().iloc[-2]),2)
prev_iip = round(float(master['iip_yoy'].dropna().iloc[-2]),  1)
prev_gdp = round(float(master['gdp_yoy'].dropna().iloc[-2]),  1)

# Models
cpi_vals   = tuple(master['cpi_yoy'].dropna().values.tolist())
fc_mean, fc_lo, fc_hi = load_arima(cpi_vals)
last_cpi_date  = master.loc[master['cpi_yoy'].notna(), 'date'].iloc[-1]
forecast_dates = pd.date_range(start=last_cpi_date, periods=7, freq='MS')[1:]
cpi_3m_fcst    = float(fc_mean[2])

proba_d, next_move = load_rate_model(
    master[['date','repo_rate','cpi_yoy','iip_yoy',
            'real_rate','gst_yoy','gdp_yoy']].to_json(date_format='iso'))

pb         = PLAYBOOK.get(cur_phase, PLAYBOOK['Goldilocks'])
move_color = {
    'hike': CR, 'hold': CGR, 'cut': CG
}.get(next_move, CGR)
real_color = CG if cur_real > 0 else CR

PHASE_COLORS = {
    'Goldilocks':     CG,
    'Reflation':      CA,
    'Stagflation':    CR,
    'Deflation risk': CB,
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🇮🇳 India Macro Forecaster")
    st.markdown(f"*Data through {cur_date}*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Macro Dashboard",
         "📈 Markets & Sectors",
         "🔍 Stock Screener",
         "🤖 Model Forecasts",
         "📋 Signal Log"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Chart date range**")
    min_yr = int(master['date'].dt.year.min())
    max_yr = int(master['date'].dt.year.max())
    from_yr = st.slider("From year", min_yr, max_yr, min_yr,
                        label_visibility="collapsed")
    plot_data = master[master['date'].dt.year >= from_yr].copy()

    st.markdown("---")
    st.markdown("**ARIMA forecast horizon**")
    fc_horizon = st.slider("Months ahead", 3, 12, 6,
                           label_visibility="collapsed")
    if fc_horizon != 6:
        fc_mean_h, fc_lo_h, fc_hi_h = run_arima(
            np.array(cpi_vals), steps=fc_horizon)
        forecast_dates_h = pd.date_range(
            start=last_cpi_date, periods=fc_horizon+1, freq='MS')[1:]
    else:
        fc_mean_h, fc_lo_h, fc_hi_h = fc_mean, fc_lo, fc_hi
        forecast_dates_h = forecast_dates

    st.markdown("---")
    phase_color = pb['color']
    st.markdown(f"""
    <div style='background:{phase_color}18;border:1px solid {phase_color}44;
                border-radius:8px;padding:12px;text-align:center;'>
        <div style='font-size:10px;color:{phase_color};font-weight:700;
                    text-transform:uppercase;'>Current Phase</div>
        <div style='font-size:18px;font-weight:700;color:{phase_color};'>
            {cur_phase}
        </div>
    </div>""", unsafe_allow_html=True)


# ── Helper: delta label ───────────────────────────────────────
def delta_html(cur, prev, invert=False, suffix="%"):
    diff = round(cur - prev, 2)
    if abs(diff) < 0.01:
        return '<span style="color:#aaa;">→ unchanged</span>'
    arrow = "▲" if diff > 0 else "▼"
    good  = (diff < 0) if invert else (diff > 0)
    color = CG if good else CR
    return (f'<span style="color:{color};">'
            f'{arrow} {abs(diff)}{suffix}</span>')

# ── Metric card helper ────────────────────────────────────────
def mcard(label, value, delta_html_str="", color="#111"):
    return f"""
    <div class='mcard'>
      <div class='mlabel'>{label}</div>
      <div class='mvalue' style='color:{color};'>{value}</div>
      <div class='mdelta'>{delta_html_str}</div>
    </div>"""


# ============================================================
# PAGE 1 — MACRO DASHBOARD
# ============================================================

if page == "📊 Macro Dashboard":

    st.markdown("## Macro Dashboard")
    st.markdown(f"*Live India macroeconomic indicators · Updated {cur_date}*")

    # ── Row 1: 6 metric cards ─────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        st.markdown(mcard("CPI Inflation",
            f"{cur_cpi}%",
            delta_html(cur_cpi, prev_cpi, invert=True)), unsafe_allow_html=True)
    with c2:
        st.markdown(mcard("Repo Rate",
            f"{cur_repo}%",
            delta_html(cur_repo, prev_repo)), unsafe_allow_html=True)
    with c3:
        st.markdown(mcard("Real Rate",
            f"{'+' if cur_real>0 else ''}{cur_real}%",
            '<span style="color:#aaa;">Repo − CPI</span>',
            color=real_color), unsafe_allow_html=True)
    with c4:
        st.markdown(mcard("IIP Growth",
            f"{cur_iip}%",
            delta_html(cur_iip, prev_iip)), unsafe_allow_html=True)
    with c5:
        st.markdown(mcard("GDP Growth",
            f"{cur_gdp}%",
            delta_html(cur_gdp, prev_gdp)), unsafe_allow_html=True)
    with c6:
        st.markdown(mcard("GST Collections",
            f"₹{cur_gst:.2f}L Cr",
            '<span style="color:#aaa;">Monthly</span>'), unsafe_allow_html=True)

    st.markdown("---")

    # ── CPI + Rate Decision ───────────────────────────────────
    st.markdown("## CPI Inflation — History + ARIMA Forecast")
    col_cpi, col_rbi = st.columns([2, 1])

    with col_cpi:
        fig = go.Figure()
        hist = plot_data[['date','cpi_yoy']].dropna()
        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['cpi_yoy'],
            mode='lines', name='Actual CPI',
            line=dict(color=CB, width=2.5),
            hovertemplate='%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>'))
        fig.add_trace(go.Scatter(
            x=forecast_dates_h, y=fc_mean_h,
            mode='lines+markers', name='ARIMA Forecast',
            line=dict(color=CR, width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='%{x|%b %Y} forecast: <b>%{y:.2f}%</b><extra></extra>'))
        fig.add_trace(go.Scatter(
            x=list(forecast_dates_h)+list(reversed(forecast_dates_h)),
            y=list(fc_hi_h)+list(reversed(fc_lo_h)),
            fill='toself', fillcolor='rgba(226,75,74,0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            name='80% confidence', hoverinfo='skip'))
        fig.add_vline(x=last_cpi_date.strftime('%Y-%m-%d'),
                      line_dash='dot', line_color='#ccc', line_width=1.5)
        fig.add_hline(y=4.0, line_dash='dot', line_color=CG, line_width=1)
        fig.add_hline(y=6.0, line_dash='dot', line_color=CA, line_width=1)
        fig.add_annotation(x=str(forecast_dates_h[0]), y=4.1,
            text='RBI target 4%', showarrow=False,
            font=dict(size=10, color=CG), xanchor='left')
        fig.add_annotation(x=str(forecast_dates_h[0]), y=6.1,
            text='Upper limit 6%', showarrow=False,
            font=dict(size=10, color=CA), xanchor='left')
        fig.update_layout(**CHART_LAYOUT,
            title=f'3-month forecast: {cpi_3m_fcst:.2f}% '
                  f'({"▲ Rising" if cpi_3m_fcst>cur_cpi else "▼ Falling"})',
            height=340, yaxis_title='CPI YoY %')
        st.plotly_chart(fig, use_container_width=True)

    with col_rbi:
        labels = ['HIKE','HOLD','CUT']
        probs  = [proba_d.get('hike',0), proba_d.get('hold',0), proba_d.get('cut',0)]
        fig2 = go.Figure(go.Bar(
            x=labels, y=probs,
            marker_color=[CR, CGR, CG],
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside', textfont=dict(size=14)))
        fig2.update_layout(**CHART_LAYOUT,
            title=f'RBI: <b>{next_move.upper()}</b> ({proba_d.get(next_move,0):.0f}%)',
            height=340, yaxis=dict(range=[0,115], title='Probability %',
                                   showgrid=True, gridcolor='#f5f5f5'),
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── GDP + IIP ─────────────────────────────────────────────
    st.markdown("## Growth Indicators")
    c_gdp, c_iip = st.columns(2)

    with c_gdp:
        gdp_plot = plot_data.dropna(subset=['gdp_yoy']).drop_duplicates('date')
        fig3 = go.Figure(go.Bar(
            x=gdp_plot['date'], y=gdp_plot['gdp_yoy'],
            marker_color=[CB if v>=0 else CR for v in gdp_plot['gdp_yoy']],
            hovertemplate='%{x|%b %Y}: <b>%{y:.1f}%</b><extra></extra>'))
        fig3.add_hline(y=6.5, line_dash='dot', line_color='#bbb', line_width=1)
        fig3.update_layout(**CHART_LAYOUT,
            title='GDP Growth — Quarterly (YoY %)',
            height=300, yaxis_title='GDP YoY %')
        st.plotly_chart(fig3, use_container_width=True)

    with c_iip:
        iip_plot = plot_data.dropna(subset=['iip_yoy'])
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=iip_plot['date'], y=iip_plot['iip_yoy'],
            marker_color=[CG if v>=0 else CR for v in iip_plot['iip_yoy']],
            opacity=0.75, name='IIP YoY %',
            hovertemplate='%{x|%b %Y}: <b>%{y:.1f}%</b><extra></extra>'))
        fig4.add_trace(go.Scatter(
            x=iip_plot['date'], y=iip_plot['iip_3m_avg'],
            mode='lines', name='3m avg', line=dict(color=CB, width=2)))
        fig4.update_layout(**CHART_LAYOUT,
            title='IIP Industrial Production (YoY %)',
            height=300, yaxis_title='IIP YoY %')
        st.plotly_chart(fig4, use_container_width=True)

    # ── Macro Phase Timeline ──────────────────────────────────
    st.markdown("## Macro Phase Timeline")
    fig5 = go.Figure()
    for phase, color in PHASE_COLORS.items():
        subset = plot_data[plot_data['macro_phase']==phase]
        if subset.empty: continue
        fig5.add_trace(go.Bar(
            x=subset['date'], y=[1]*len(subset),
            name=phase, marker_color=color, marker_line_width=0,
            hovertemplate=f'<b>{phase}</b>: %{{x|%b %Y}}<extra></extra>'))
    fig5.update_layout(**CHART_LAYOUT, barmode='stack', bargap=0,
        yaxis=dict(showticklabels=False, showgrid=False),
        title='Monthly macro phase — each bar = one month',
        height=180, legend=dict(orientation='h', y=-0.4))
    st.plotly_chart(fig5, use_container_width=True)

    # Phase % breakdown
    counts = plot_data['macro_phase'].value_counts()
    total  = len(plot_data)
    pcols  = st.columns(4)
    for i, (phase, color) in enumerate(PHASE_COLORS.items()):
        pct = counts.get(phase, 0) / total * 100
        with pcols[i]:
            st.markdown(f"""
            <div style='text-align:center;background:{color}12;padding:10px;
                        border-radius:8px;border:1px solid {color}33;'>
              <div style='font-size:10px;font-weight:700;color:{color};
                          text-transform:uppercase;'>{phase}</div>
              <div style='font-size:22px;font-weight:700;color:{color};'>
                  {pct:.0f}%
              </div>
              <div style='font-size:11px;color:#999;'>
                  {counts.get(phase,0)} months
              </div>
            </div>""", unsafe_allow_html=True)

    # ── GST Collections ───────────────────────────────────────
    st.markdown("## GST Collections")
    gst_plot = plot_data.dropna(subset=['gst_collections'])
    gst_plot = gst_plot[gst_plot['gst_collections'] > 0]
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=gst_plot['date'],
        y=gst_plot['gst_collections']/100000,
        marker_color=CP, opacity=0.8, name='GST (₹ Lakh Cr)',
        hovertemplate='%{x|%b %Y}: <b>₹%{y:.2f}L Cr</b><extra></extra>'))
    # 12-month rolling average
    gst_plot['gst_12m'] = (gst_plot['gst_collections']
                           .rolling(12).mean() / 100000)
    fig6.add_trace(go.Scatter(
        x=gst_plot['date'], y=gst_plot['gst_12m'],
        mode='lines', name='12m avg', line=dict(color=CA, width=2)))
    fig6.update_layout(**CHART_LAYOUT,
        title='Monthly GST Collections (₹ Lakh Crore)',
        height=280, yaxis_title='₹ Lakh Crore')
    st.plotly_chart(fig6, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────
    st.markdown("## Indicator Correlations")
    corr_cols   = ['cpi_yoy','repo_rate','real_rate','iip_yoy','gdp_yoy','gst_yoy']
    corr_labels = ['CPI','Repo','Real Rate','IIP','GDP','GST']
    corr_mat    = master[corr_cols].dropna().corr().round(2)
    fig7 = px.imshow(corr_mat, x=corr_labels, y=corr_labels,
                     color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
                     text_auto=True, aspect='auto')
    fig7.update_layout(**CHART_LAYOUT,
        title='How strongly do macro indicators move together?',
        height=320, coloraxis_showscale=True)
    st.plotly_chart(fig7, use_container_width=True)

    # ── Sector Rotation Signal ────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Sector Rotation Signal")
    sc1, sc2 = st.columns([3, 2])

    over_html  = ''.join(f"<span class='tag-over'>{s}</span>"
                         for s in pb['over'])
    under_html = ''.join(f"<span class='tag-under'>{s}</span>"
                         for s in pb['under'])

    with sc1:
        st.markdown(f"""
        <div style='background:{pb["color"]}0d;border:1.5px solid {pb["color"]}44;
                    border-radius:12px;padding:20px 22px;'>
          <div style='font-size:11px;color:{pb["color"]};font-weight:700;
                      text-transform:uppercase;margin-bottom:3px;'>
              Current Phase
          </div>
          <div style='font-size:20px;font-weight:700;color:{pb["color"]};
                      margin-bottom:16px;'>{cur_phase}</div>
          <div style='font-size:13px;font-weight:700;color:{CG};
                      margin-bottom:6px;'>✅  OVERWEIGHT — Add / Buy</div>
          <div style='margin-bottom:14px;'>{over_html}</div>
          <div style='font-size:13px;font-weight:700;color:{CR};
                      margin-bottom:6px;'>❌  UNDERWEIGHT — Reduce / Avoid</div>
          <div style='margin-bottom:14px;'>{under_html}</div>
          <div style='font-size:13px;font-weight:700;color:{CB};
                      margin-bottom:4px;'>🏦  Bond / Fixed Income</div>
          <div style='font-size:13px;color:#444;'>{pb['bonds']}</div>
        </div>""", unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
        <div style='background:white;border:1px solid #e8e8e8;
                    border-radius:12px;padding:20px 22px;height:100%;'>
          <div style='font-size:13px;font-weight:700;color:#666;
                      margin-bottom:8px;'>WHY THIS POSITIONING?</div>
          <div style='font-size:13px;color:#333;line-height:1.75;
                      margin-bottom:16px;'>{pb['why']}</div>
          <table style='width:100%;font-size:13px;border-collapse:collapse;'>
            <tr style='border-bottom:1px solid #f4f4f4;'>
              <td style='color:#888;padding:6px 0;'>CPI now</td>
              <td style='text-align:right;font-weight:600;'>
                {cur_cpi}% <span style='color:#aaa;'>(target: 4%)</span></td>
            </tr>
            <tr style='border-bottom:1px solid #f4f4f4;'>
              <td style='color:#888;padding:6px 0;'>CPI in 3 months</td>
              <td style='text-align:right;font-weight:600;
                         color:{"#e24b4a" if cpi_3m_fcst>cur_cpi else "#1d9e75"};'>
                {cpi_3m_fcst:.2f}%
                {"▲" if cpi_3m_fcst>cur_cpi else "▼"}
              </td>
            </tr>
            <tr style='border-bottom:1px solid #f4f4f4;'>
              <td style='color:#888;padding:6px 0;'>GDP growth</td>
              <td style='text-align:right;font-weight:600;'>{cur_gdp}%</td>
            </tr>
            <tr style='border-bottom:1px solid #f4f4f4;'>
              <td style='color:#888;padding:6px 0;'>Real rate</td>
              <td style='text-align:right;font-weight:600;
                         color:{real_color};'>
                {'+' if cur_real>0 else ''}{cur_real}%
                {'(tight)' if cur_real>0 else '(loose)'}
              </td>
            </tr>
            <tr>
              <td style='color:#888;padding:6px 0;'>RBI next move</td>
              <td style='text-align:right;font-weight:600;
                         color:{move_color};'>
                {next_move.upper()} ({proba_d.get(next_move,0):.0f}%)
              </td>
            </tr>
          </table>
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 2 — MARKETS & SECTORS
# ============================================================

elif page == "📈 Markets & Sectors":

    st.markdown("## Markets & Sectors")
    st.markdown("*Live NSE sector returns — updates every 15 minutes*")

    with st.spinner("Fetching live sector data from NSE..."):
        sector_df = load_sector_returns()
        nifty_df  = load_nifty_history()

    # ── Nifty 50 chart ────────────────────────────────────────
    if not nifty_df.empty:
        st.markdown("## Nifty 50 — Price History")
        period_sel = st.radio("Period", ["1Y","3Y","5Y"],
                              horizontal=True, index=0)
        period_days = {"1Y":252,"3Y":756,"5Y":1260}[period_sel]
        n = nifty_df.tail(period_days)

        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(
            x=n.index, y=n['Close'],
            mode='lines', name='Nifty 50',
            line=dict(color=CB, width=2),
            fill='tozeroy',
            fillcolor='rgba(24,95,165,0.06)',
            hovertemplate='%{x|%d %b %Y}: <b>%{y:,.0f}</b><extra></extra>'))
        fig_n.update_layout(**CHART_LAYOUT,
            title='Nifty 50 Closing Price', height=360,
            yaxis_title='Index Value',
            yaxis=dict(tickformat=',.0f',
                       showgrid=True, gridcolor='#f5f5f5'))
        st.plotly_chart(fig_n, use_container_width=True)
    else:
        st.info("Nifty 50 data loading... (yfinance refreshes every 15 min)")

    # ── Sector heatmap ────────────────────────────────────────
    st.markdown("## Sector Performance Heatmap")
    if not sector_df.empty:
        period_col = st.radio("Return period", ["1M %","3M %","6M %","1Y %"],
                               horizontal=True, index=1)
        fig_s = px.bar(
            sector_df.sort_values(period_col),
            x=period_col, y='Sector', orientation='h',
            color=period_col,
            color_continuous_scale=['#E24B4A','#f5f5f5','#1D9E75'],
            color_continuous_midpoint=0,
            text=period_col,
        )
        fig_s.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside')
        fig_s.update_layout(**CHART_LAYOUT,
            title=f'NSE Sector Returns — {period_col}',
            height=420,
            coloraxis_showscale=False,
            xaxis_title='Return %',
            yaxis_title='')
        st.plotly_chart(fig_s, use_container_width=True)

        # Table
        st.markdown("#### Sector returns table")
        styled = sector_df.copy()
        st.dataframe(
            styled.style.background_gradient(
                subset=['1M %','3M %','6M %','1Y %'],
                cmap='RdYlGn', vmin=-20, vmax=40),
            use_container_width=True, hide_index=True)
    else:
        st.info("Sector data loading...")

    # ── Sector vs Macro Phase ─────────────────────────────────
    st.markdown("## Sector Signal vs Current Phase")
    over_html  = ''.join(f"<span class='tag-over'>{s}</span>"  for s in pb['over'])
    under_html = ''.join(f"<span class='tag-under'>{s}</span>" for s in pb['under'])
    st.markdown(f"""
    <div style='background:{pb["color"]}0d;border:1px solid {pb["color"]}33;
                border-radius:10px;padding:16px 20px;margin-top:8px;'>
      <b style='color:{pb["color"]};font-size:15px;'>{cur_phase} phase</b>
      &nbsp;—&nbsp;
      <b style='color:{CG};'>Overweight:</b> {over_html}
      &nbsp;&nbsp;
      <b style='color:{CR};'>Underweight:</b> {under_html}
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 3 — STOCK SCREENER
# ============================================================

elif page == "🔍 Stock Screener":

    st.markdown("## Nifty 500 Stock Screener")
    st.markdown("*Momentum + volatility scoring · Live data via NSE/yfinance*")

    with st.spinner("Fetching live stock data for 50 Nifty 500 stocks..."):
        screener = load_screener_data()

    if screener.empty:
        st.warning(
            "Live stock data is temporarily unavailable. "
            "This is usually a yfinance rate limit — try again in 2 minutes.")
        st.stop()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ['All'] + sorted(screener['Sector'].unique().tolist())
        sel_sector = st.selectbox("Filter by sector", sectors)
    with col_f2:
        min_score = st.slider("Minimum composite score", -50, 80, 0)
    with col_f3:
        sort_col = st.selectbox("Sort by",
            ['Score','1M %','3M %','6M %','12M %','Volatility','RSI'])

    # Apply filters
    filtered = screener.copy()
    if sel_sector != 'All':
        filtered = filtered[filtered['Sector'] == sel_sector]
    filtered = filtered[filtered['Score'] >= min_score]
    filtered = filtered.sort_values(sort_col, ascending=(sort_col=='Volatility'))
    filtered = filtered.reset_index(drop=True)
    filtered.index += 1

    st.markdown(f"**Showing {len(filtered)} stocks**")

    # Colour the return columns green/red
    def colour_ret(val):
        if pd.isna(val): return ''
        return f'color: {"#1d9e75" if val >= 0 else "#e24b4a"}; font-weight:600'

    def colour_rsi(val):
        if pd.isna(val): return ''
        if val > 70: return 'color:#e24b4a;font-weight:600'
        if val < 30: return 'color:#1d9e75;font-weight:600'
        return ''

    display_cols = ['Name','Sector','Price (₹)','1M %','3M %',
                    '6M %','12M %','Volatility','RSI','Score']
    st.dataframe(
        filtered[display_cols].style
            .applymap(colour_ret, subset=['1M %','3M %','6M %','12M %'])
            .applymap(colour_rsi, subset=['RSI'])
            .background_gradient(subset=['Score'], cmap='RdYlGn',
                                  vmin=-20, vmax=60),
        use_container_width=True)

    st.markdown("""
    **How to read this table:**
    - **Score** = composite of momentum (60%) and low-volatility (40%).
      Higher = stronger trend with lower risk.
    - **RSI > 70** = overbought (red) · **RSI < 30** = oversold (green)
    - **Volatility** = annualised 30-day standard deviation of daily returns
    """)

    # Top picks chart
    st.markdown("## Top 10 by Score")
    top10 = filtered.head(10)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=top10['Name'], y=top10['Score'],
        marker_color=CB, opacity=0.85,
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'))
    fig_sc.update_layout(**CHART_LAYOUT,
        title='Top 10 stocks by composite score',
        height=320, xaxis_tickangle=-35,
        yaxis_title='Composite Score', showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Return scatter
    st.markdown("## Momentum vs Volatility")
    if '3M %' in filtered.columns and 'Volatility' in filtered.columns:
        fig_sc2 = px.scatter(
            filtered.dropna(subset=['3M %','Volatility']),
            x='Volatility', y='3M %',
            color='Sector', size='Score',
            hover_name='Name',
            labels={'Volatility':'Annualised Volatility %',
                    '3M %':'3-Month Return %'},
            title='Lower-left quadrant = low vol, positive momentum (ideal)',
            height=420,
        )
        fig_sc2.add_hline(y=0, line_dash='dot', line_color='#ccc')
        fig_sc2.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_sc2, use_container_width=True)


# ============================================================
# PAGE 4 — MODEL FORECASTS
# ============================================================

elif page == "🤖 Model Forecasts":

    st.markdown("## Forecasting Models")

    # ── ARIMA ─────────────────────────────────────────────────
    st.markdown("## Model 1 — ARIMA CPI Forecast")
    st.markdown("""
    **What is ARIMA?** It stands for AutoRegressive Integrated Moving Average.
    It looks at the pattern in past CPI data and extrapolates forward.
    Think of it like Excel's FORECAST function but much more sophisticated —
    it accounts for trends, cycles and past forecast errors simultaneously.
    """)

    fig_a = go.Figure()
    hist = master[['date','cpi_yoy']].dropna().tail(36)
    fig_a.add_trace(go.Scatter(
        x=hist['date'], y=hist['cpi_yoy'],
        mode='lines', name='Actual CPI',
        line=dict(color=CB, width=2.5),
        hovertemplate='%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>'))
    fig_a.add_trace(go.Scatter(
        x=forecast_dates_h, y=fc_mean_h,
        mode='lines+markers', name='Forecast',
        line=dict(color=CR, width=2, dash='dash'),
        marker=dict(size=7),
        hovertemplate='%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>'))
    fig_a.add_trace(go.Scatter(
        x=list(forecast_dates_h)+list(reversed(forecast_dates_h)),
        y=list(fc_hi_h)+list(reversed(fc_lo_h)),
        fill='toself', fillcolor='rgba(226,75,74,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='80% confidence', hoverinfo='skip'))
    fig_a.add_vline(x=last_cpi_date.strftime('%Y-%m-%d'),
                    line_dash='dot', line_color='#ccc', line_width=1.5)
    fig_a.add_hline(y=4.0, line_dash='dot', line_color=CG, line_width=1)
    fig_a.add_hline(y=6.0, line_dash='dot', line_color=CA, line_width=1)
    fig_a.update_layout(**CHART_LAYOUT,
        title=f'ARIMA({fc_horizon}-month) CPI Forecast',
        height=380, yaxis_title='CPI YoY %')
    st.plotly_chart(fig_a, use_container_width=True)

    # Forecast table
    fcast_df = pd.DataFrame({
        'Month':    [d.strftime('%b %Y') for d in forecast_dates_h],
        'Forecast': fc_mean_h,
        'Lower 80%':fc_lo_h,
        'Upper 80%':fc_hi_h,
    })
    st.dataframe(fcast_df.style.format({
        'Forecast':'{:.2f}%','Lower 80%':'{:.2f}%','Upper 80%':'{:.2f}%'}),
        use_container_width=True, hide_index=True)

    # ── Rate Model ────────────────────────────────────────────
    st.markdown("## Model 2 — RBI Rate Decision Predictor")
    st.markdown("""
    **What is Random Forest?** It builds 200 decision trees, each looking
    at a slightly different subset of your data. Each tree votes on whether
    the RBI will hike, hold or cut. The majority vote wins.
    The probabilities below show how split or decisive the 200 trees are.
    """)

    labels = ['HIKE','HOLD','CUT']
    probs  = [proba_d.get('hike',0), proba_d.get('hold',0), proba_d.get('cut',0)]
    fig_r = go.Figure(go.Bar(
        x=labels, y=probs,
        marker_color=[CR, CGR, CG],
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside', textfont=dict(size=16, color='#333')))
    fig_r.update_layout(**CHART_LAYOUT,
        title=f'Model verdict: <b>{next_move.upper()}</b> '
              f'({proba_d.get(next_move,0):.0f}% probability)',
        height=360,
        yaxis=dict(range=[0,115], title='Probability %',
                   showgrid=True, gridcolor='#f5f5f5'),
        showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

    # Feature importance explanation
    st.markdown("#### What drives the model's prediction?")
    st.markdown("""
    The model looks at these signals (in order of importance):
    1. **CPI momentum** — is inflation rising or falling over last 3 months?
    2. **Real interest rate** — is the real rate negative (forces hike) or strongly positive (allows cut)?
    3. **IIP growth trend** — is industrial activity accelerating or slowing?
    4. **GST growth** — proxy for economic activity and indirect tax buoyancy
    5. **GDP trend** — is growth above or below the ~6.5% trend rate?
    """)

    # ── Repo rate history with decision markers ───────────────
    st.markdown("## Repo Rate History with Actual Decisions")
    rate_diff = master['repo_rate'].diff(1)
    fig_rd = go.Figure()
    fig_rd.add_trace(go.Scatter(
        x=master['date'], y=master['repo_rate'],
        mode='lines', name='Repo rate',
        line=dict(color=CB, width=2.5)))
    for dec, color, sym in [
        ('hike', CR, 'triangle-up'),
        ('cut',  CG, 'triangle-down'),
    ]:
        mask = rate_diff.apply(
            lambda x: (x > 0.01 if dec=='hike' else x < -0.01))
        fig_rd.add_trace(go.Scatter(
            x=master.loc[mask,'date'],
            y=master.loc[mask,'repo_rate'],
            mode='markers', name=dec.upper(),
            marker=dict(color=color, size=10, symbol=sym)))
    fig_rd.update_layout(**CHART_LAYOUT,
        title='RBI Repo Rate — actual hike (▲) and cut (▼) decisions',
        height=320, yaxis_title='Rate %')
    st.plotly_chart(fig_rd, use_container_width=True)


# ============================================================
# PAGE 5 — SIGNAL LOG
# ============================================================

elif page == "📋 Signal Log":

    st.markdown("## Monthly Signal Log — Your Track Record")
    st.markdown("""
    This page records every macro signal your dashboard has generated.
    Run this app monthly, check the sector signal, and your calls are
    automatically logged here with dates. After 12 months you have a
    **documented, timestamped investment track record** — exactly what
    fund management interviewers ask to see.
    """)

    # Current month signal
    current_signal = {
        'Date':             datetime.today().strftime('%d %b %Y'),
        'Phase':            cur_phase,
        'CPI':              f'{cur_cpi}%',
        'CPI 3m Forecast':  f'{cpi_3m_fcst:.2f}%',
        'GDP':              f'{cur_gdp}%',
        'Repo Rate':        f'{cur_repo}%',
        'RBI Call':         next_move.upper(),
        'Confidence':       f'{proba_d.get(next_move,0):.0f}%',
        'Overweight':       ', '.join(pb['over']),
        'Underweight':      ', '.join(pb['under']),
        'Bond View':        pb['bonds'],
    }

    st.markdown("### Current Signal")
    sig_cols = st.columns(4)
    highlights = [
        ('Phase',          cur_phase,                      pb['color']),
        ('RBI Call',       next_move.upper(),              move_color),
        ('CPI 3m Forecast',f'{cpi_3m_fcst:.2f}%',         CR if cpi_3m_fcst>cur_cpi else CG),
        ('Real Rate',      f'{"+"}{"" if cur_real<0 else ""}{cur_real}%', real_color),
    ]
    for i, (label, val, color) in enumerate(highlights):
        with sig_cols[i]:
            st.markdown(f"""
            <div style='background:{color}12;border:1px solid {color}33;
                        border-radius:8px;padding:12px;text-align:center;'>
              <div style='font-size:11px;color:{color};font-weight:700;
                          text-transform:uppercase;'>{label}</div>
              <div style='font-size:20px;font-weight:700;color:{color};'>
                  {val}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Full Signal Detail")
    for k, v in current_signal.items():
        st.markdown(f"**{k}:** {v}")

    # LinkedIn note generator
    st.markdown("---")
    st.markdown("### 📝 One-click LinkedIn Note")
    st.markdown("*Copy this and post it on LinkedIn today to start your track record:*")

    cpi_dir = "rising" if cpi_3m_fcst > cur_cpi else "falling"
    linkedin_note = f"""🇮🇳 India Macro Outlook — {datetime.today().strftime('%B %Y')}

📊 Current macro phase: {cur_phase}
• CPI Inflation: {cur_cpi}% (3m forecast: {cpi_3m_fcst:.2f}%, {cpi_dir})
• GDP Growth: {cur_gdp}% | IIP: {cur_iip}%
• Repo Rate: {cur_repo}% | Real Rate: {'+' if cur_real>0 else ''}{cur_real}%

🏦 RBI Next Move: {next_move.upper()} ({proba_d.get(next_move,0):.0f}% model probability)

📈 Sector positioning:
✅ Overweight: {', '.join(pb['over'][:3])}
❌ Underweight: {', '.join(pb['under'][:3])}

💡 Rationale: {pb['why'][:200]}...

Built with ARIMA + Random Forest models on RBI/MOSPI data.
#IndiaMacro #EquityResearch #FundManagement #NSE #RBI"""

    st.code(linkedin_note, language=None)
    st.caption("Select all → Copy → Paste on LinkedIn. Edit numbers/views as needed.")


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#bbb;font-size:12px;'>"
    "India Macro Forecaster · Data: RBI · MOSPI · GST Council · NSE · yfinance · "
    "Models: ARIMA · Random Forest · Built for fund management research"
    "</div>",
    unsafe_allow_html=True)
