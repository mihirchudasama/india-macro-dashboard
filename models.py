# ============================================================
# models.py  —  ALL FORECASTING MODELS
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def run_arima(series_values: np.ndarray, steps: int = 6):
    """
    Fits ARIMA(2,1,2) on a 1D numpy array.
    Returns (forecast_mean, lower_80, upper_80) as numpy arrays.
    """
    try:
        fit  = ARIMA(series_values, order=(2, 1, 2)).fit()
        fc   = fit.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci   = fc.conf_int(alpha=0.20)
        mean = mean.values if hasattr(mean, 'values') else np.array(mean, dtype=float)
        if hasattr(ci, 'iloc'):
            lo = ci.iloc[:, 0].values.astype(float)
            hi = ci.iloc[:, 1].values.astype(float)
        else:
            arr = np.array(ci, dtype=float)
            lo, hi = arr[:, 0], arr[:, 1]
        return np.round(mean, 2), np.round(lo, 2), np.round(hi, 2)
    except Exception:
        last = float(series_values[-1])
        flat = np.full(steps, last)
        return flat, flat - 0.5, flat + 0.5


def run_rate_model(master: pd.DataFrame):
    """
    Trains a Random Forest classifier to predict
    RBI's next rate decision: hike / hold / cut.
    Returns (proba_dict, predicted_decision).
    """
    try:
        df = master[['repo_rate','cpi_yoy','iip_yoy',
                     'real_rate','gst_yoy','gdp_yoy']].dropna().copy()

        base_cols = ['cpi_yoy','iip_yoy','real_rate','gst_yoy','gdp_yoy']
        for col in base_cols:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
        df['cpi_above'] = (df['cpi_yoy'] > 6.0).astype(int)
        df['neg_real']  = (df['real_rate'] < 0.0).astype(int)
        df['decision']  = df['repo_rate'].diff(1).apply(
            lambda x: 'hike' if x > 0.01 else ('cut' if x < -0.01 else 'hold'))
        df = df.dropna().reset_index(drop=True)

        feat_cols = [c for c in df.columns
                     if 'lag' in c or c in ['cpi_above', 'neg_real']]
        X, y  = df[feat_cols], df['decision']
        split = int(len(X) * 0.8)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=4,
            class_weight='balanced', random_state=42)
        clf.fit(X[:split], y[:split])

        raw_p   = clf.predict_proba(X.iloc[[-1]])[0]
        proba_d = {c: round(float(p)*100, 1)
                   for c, p in zip(clf.classes_, raw_p)}
        decision = max(proba_d, key=proba_d.get)
        return proba_d, decision
    except Exception:
        return {'hike': 33.0, 'hold': 34.0, 'cut': 33.0}, 'hold'


PLAYBOOK = {
    'Goldilocks': {
        'over' : ['IT / Technology', 'Private Banks', 'Auto',
                  'Capital Goods', 'FMCG / Consumption'],
        'under': ['Metals / Mining', 'Oil & Gas', 'Pharma'],
        'bonds': 'Neutral — growth supportive, rates stable',
        'why'  : ('High growth + low inflation = corporate earnings expanding, '
                  'consumers spending freely, credit quality good. '
                  'Growth and quality stocks lead the market.'),
        'color': '#1D9E75',
    },
    'Reflation': {
        'over' : ['Metals & Mining', 'Oil & Gas', 'PSU Banks',
                  'Infrastructure / Cement', 'Capital Goods'],
        'under': ['IT / Technology', 'FMCG / Consumption', 'NBFCs'],
        'bonds': 'Negative — inflation premium pushes bond yields higher',
        'why'  : ('Growth with rising prices. Commodity producers and banks '
                  'with pricing power outperform. Tech and consumer '
                  'discretionary lag as input costs rise.'),
        'color': '#EF9F27',
    },
    'Stagflation': {
        'over' : ['FMCG / Staples', 'Pharma / Healthcare',
                  'Utilities', 'Gold / Precious Metals'],
        'under': ['Auto', 'Capital Goods', 'Real Estate',
                  'Private Banks', 'IT'],
        'bonds': 'Very negative — gold is the real safe haven here',
        'why'  : ('Worst macro phase. Defensive positioning critical. '
                  'Only non-discretionary essentials hold up. '
                  'Avoid all cyclicals.'),
        'color': '#E24B4A',
    },
    'Deflation risk': {
        'over' : ['Long-duration Bonds / Gilt Funds',
                  'Rate-sensitive NBFCs / HFCs', 'Pharma', 'FMCG'],
        'under': ['Metals', 'PSU Banks', 'Commodities', 'Capital Goods'],
        'bonds': 'Very positive — RBI will cut, long-duration bonds rally hard',
        'why'  : ('RBI will cut rates. Bond and gilt funds rally strongly. '
                  'Rate-sensitive sectors re-rate upward. '
                  'Commodities and PSU banks suffer.'),
        'color': '#185FA5',
    },
}
