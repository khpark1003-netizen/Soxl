# -*- coding: utf-8 -*-
"""지표 계산 + 통계 함수 (원본 역탐색 v6 로직 그대로)"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import binomtest


# ---------- RSI ----------
def rsi_wilders(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_cutler(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------- MACD ----------
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


# ---------- ATR / ADX ----------
def _true_range(high, low, close):
    return pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)


def calc_atr(high, low, close, period=14):
    return _true_range(high, low, close).rolling(period).mean()


def calc_adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _true_range(high, low, close).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di


# ---------- 볼린저 %B ----------
def bollinger(series, period=20, std_k=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_k * std
    lower = ma - std_k * std
    return (series - lower) / (upper - lower + 1e-9)


# ---------- RSI 다이버전스 ----------
def detect_divergence(close, rsi, lookback=20):
    if len(close) < lookback:
        return None
    c = close.iloc[-lookback:]
    r = rsi.iloc[-lookback:]
    half = lookback // 2
    c_prev = close.iloc[-lookback:-half]
    r_prev = rsi.iloc[-lookback:-half]
    if len(c_prev) < 3:
        return None
    bull_div = (c.min() < c_prev.min()) and (r.min() > r_prev.min())
    bear_div = (c.max() > c_prev.max()) and (r.max() < r_prev.max())
    if bull_div:
        return "상승 다이버전스"
    elif bear_div:
        return "하락 다이버전스"
    return None


# ---------- 성과 통계 ----------
def calc_mdd(r):
    if len(r) == 0:
        return np.nan
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    return ((equity - peak) / peak).min() * 100


def calc_ev(r):
    """기대값 = 승률×평균이익 − 패율×평균손실 (%)

    원본은 전승/전패 케이스에서 NaN을 반환했다. 전승(손실 없음)은 가장 좋은
    케이스인데 오히려 NaN이 되어 탐색에서 탈락하거나 idxmax가 죽는 버그가 있었다.
    → 손실 항이 0인 것으로 계산해 그대로 평가한다.
    """
    r = pd.Series(r)
    if len(r) == 0:
        return np.nan
    wins = r[r > 0]
    losses = r[r <= 0]
    wr = len(wins) / len(r)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else 0.0
    return round((wr * avg_win - (1 - wr) * avg_loss) * 100, 3)


def sig_label(n_win, n_total, p0=0.5):
    if n_total < 5:
        return "-"
    pv = binomtest(n_win, n_total, p=p0, alternative="greater").pvalue
    if pv < 0.01:
        return "★★ p<0.01"
    elif pv < 0.05:
        return "★ p<0.05"
    return f"n.s. p={pv:.2f}"


# ---------- 데이터 ----------
def get_data(ticker, start):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()
