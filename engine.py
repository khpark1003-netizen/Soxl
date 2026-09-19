# -*- coding: utf-8 -*-
"""분석 엔진: 지표 계산 → 현재 상태 → 조합 탐색"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from indicators import (
    rsi_wilders, rsi_cutler, calc_macd, calc_atr, calc_adx, bollinger,
    detect_divergence, calc_mdd, calc_ev, sig_label, get_data,
)

BB_LABELS = ["BB하단", "BB중하단", "BB중상단", "BB상단"]
DD_LABELS = ["급락(-15%↓)", "큰낙폭(-15~-7%)", "중낙폭(-7~-3%)", "소낙폭(-3~-1%)", "고점근처"]
ATR_LABELS = ["낮음", "보통", "높음"]
ADX_LABELS = ["횡보", "약한추세", "강한추세"]
HOLD_RANGE = range(3, 21)
SHOW_HOLDS = [3, 5, 7, 10, 12, 15, 18, 20]
COND_COLS = ["RSI구간", "추세", "BB위치", "낙폭구간", "MACD방향",
             "ATR변동성", "ADX추세강도", "장기추세", "시장레짐"]


def _market_regime(sr, vz):
    if pd.isna(sr) or vz in ("nan", "None"):
        return np.nan
    if sr == "상승장" and vz in ("안정", "경계"):
        return "안정장세"
    if sr == "하락장" or vz in ("공포", "패닉"):
        return "위험장세"
    return "혼조장세"


def load_and_build(years, ticker="SOXL"):
    """데이터 로드 + 전 지표 계산. (data, close, ctx) 반환"""
    start = datetime.today() - timedelta(days=365 * years)
    data = get_data(ticker, start)
    if data.empty:
        raise RuntimeError(f"{ticker} 데이터 로드 실패")

    try:
        vix_raw = get_data("^VIX", start)
        vix_now = float(vix_raw["Close"].iloc[-1])
    except Exception:
        vix_raw, vix_now = None, np.nan
    try:
        spy_raw = get_data("SPY", start)
        spy_now = float(spy_raw["Close"].iloc[-1])
        spy_ma200_now = float(spy_raw["Close"].rolling(200).mean().iloc[-1])
    except Exception:
        spy_raw, spy_now, spy_ma200_now = None, np.nan, np.nan

    close, high, low = data["Close"], data["High"], data["Low"]
    data = data.copy()

    data["RSI_W"] = rsi_wilders(close)
    data["Sig_W"] = data["RSI_W"].rolling(6).mean()
    data["RSI_C"] = rsi_cutler(close)
    data["Sig_C"] = data["RSI_C"].rolling(6).mean()
    data["MA10"] = close.rolling(10).mean()
    data["MA30"] = close.rolling(30).mean()
    data["MA_gap_pct"] = (data["MA10"] - data["MA30"]) / data["MA30"] * 100
    data["bb_pctb"] = bollinger(close)
    data["dd_from_high"] = (close / close.rolling(20).max() - 1) * 100

    macd_line, macd_sig, macd_hist = calc_macd(close)
    data["MACD"], data["MACD_sig"], data["MACD_hist"] = macd_line, macd_sig, macd_hist
    data["ATR_ratio"] = calc_atr(high, low, close) / close * 100

    adx_val, plus_di, minus_di = calc_adx(high, low, close)
    data["ADX"], data["plus_di"], data["minus_di"] = adx_val, plus_di, minus_di

    bb_ma, bb_std = close.rolling(20).mean(), close.rolling(20).std()
    data["bb_upper"] = bb_ma + 2 * bb_std
    data["bb_lower"] = bb_ma - 2 * bb_std
    data["bb_mid"] = bb_ma

    # 장기추세
    data["MA200"] = close.rolling(200).mean()
    lt = pd.Series(np.where(close > data["MA200"], "장기상승", "장기하락"),
                   index=data.index, dtype=object)
    data["long_trend"] = lt.mask(data["MA200"].isna(), np.nan)

    # 시장 레짐
    if spy_raw is not None:
        sc = spy_raw["Close"]
        sma200 = sc.rolling(200).mean()
        spy_regime = pd.Series(np.where(sc > sma200, "상승장", "하락장"),
                               index=sc.index, dtype=object).mask(sma200.isna(), np.nan)
    else:
        spy_regime = pd.Series(dtype=object)
    if vix_raw is not None:
        vix_zone = pd.cut(vix_raw["Close"], bins=[-999, 20, 30, 40, 999],
                          labels=["안정", "경계", "공포", "패닉"])
    else:
        vix_zone = pd.Series(dtype=object)

    data["spy_regime"] = spy_regime.reindex(data.index).ffill()
    data["vix_zone"] = vix_zone.reindex(data.index).ffill().astype(str)
    data["market_regime"] = [_market_regime(s, v)
                             for s, v in zip(data["spy_regime"], data["vix_zone"])]

    data = data.dropna(subset=[
        "RSI_W", "Sig_W", "RSI_C", "Sig_C", "MA10", "MA30", "bb_pctb",
        "dd_from_high", "MACD", "MACD_sig", "ATR_ratio", "ADX",
        "MA200", "long_trend", "market_regime",
    ])

    # RSI 구간 (분위수 기반)
    pcts = [0, 15, 30, 45, 55, 70, 85, 100]
    rsi_edges = np.unique(np.round(np.percentile(data["RSI_W"].values, pcts), 1))
    rsi_labels = [f"RSI {rsi_edges[i]:.0f}~{rsi_edges[i+1]:.0f}"
                  for i in range(len(rsi_edges) - 1)]

    data["RSI_zone"] = pd.cut(data["RSI_W"], bins=rsi_edges, labels=rsi_labels,
                              include_lowest=True).astype(str)
    data["bb_zone"] = pd.cut(data["bb_pctb"], bins=[-999, .2, .5, .8, 999],
                             labels=BB_LABELS).astype(str)
    data["dd_zone"] = pd.cut(data["dd_from_high"], bins=[-999, -15, -7, -3, -1, .1],
                             labels=DD_LABELS).astype(str)
    data["atr_zone"] = pd.cut(data["ATR_ratio"], bins=[-999, 3, 6, 999],
                              labels=ATR_LABELS).astype(str)
    data["adx_zone"] = pd.cut(data["ADX"], bins=[-999, 20, 40, 999],
                              labels=ADX_LABELS).astype(str)
    data["trend"] = np.where(data["MA10"] > data["MA30"], "추세상승", "추세하락")
    data["macd_dir"] = np.where(data["MACD"] > data["MACD_sig"], "MACD상승", "MACD하락")

    close = close.reindex(data.index)
    ctx = {"vix_now": vix_now, "spy_now": spy_now, "spy_ma200_now": spy_ma200_now,
           "rsi_labels": rsi_labels, "rsi_edges": rsi_edges}
    return data, close, ctx


def current_state(data, close, ctx):
    """현재(마지막 행) 상태 딕셔너리"""
    last = data.iloc[-1]
    price = float(close.iloc[-1])
    vix_now, spy_now, spy_ma200_now = ctx["vix_now"], ctx["spy_now"], ctx["spy_ma200_now"]

    cur_rsi_zone = None
    edges, labels = ctx["rsi_edges"], ctx["rsi_labels"]
    for i in range(len(edges) - 1):
        if edges[i] <= last["RSI_W"] <= edges[i + 1]:
            cur_rsi_zone = labels[i]
            break

    bb = pd.cut([last["bb_pctb"]], bins=[-999, .2, .5, .8, 999], labels=BB_LABELS)[0]
    dd = pd.cut([last["dd_from_high"]], bins=[-999, -15, -7, -3, -1, .1], labels=DD_LABELS)[0]
    atr = pd.cut([last["ATR_ratio"]], bins=[-999, 3, 6, 999], labels=ATR_LABELS)[0]
    adx = pd.cut([last["ADX"]], bins=[-999, 20, 40, 999], labels=ADX_LABELS)[0]

    spy_regime = ("상승장" if (not np.isnan(spy_now) and not np.isnan(spy_ma200_now)
                             and spy_now > spy_ma200_now) else "하락장")
    vz_cat = pd.cut([vix_now], bins=[-999, 20, 30, 40, 999],
                    labels=["안정", "경계", "공포", "패닉"])[0]
    vix_zone = str(vz_cat) if not pd.isna(vz_cat) else "안정"
    if spy_regime == "상승장" and vix_zone in ("안정", "경계"):
        regime = "안정장세"
    elif spy_regime == "하락장" or vix_zone in ("공포", "패닉"):
        regime = "위험장세"
    else:
        regime = "혼조장세"

    return {
        "price": price,
        "rsi_w": last["RSI_W"], "sig_w": last["Sig_W"],
        "rsi_c": last["RSI_C"], "sig_c": last["Sig_C"],
        "ma10": last["MA10"], "ma30": last["MA30"], "magap": last["MA_gap_pct"],
        "bbpctb": last["bb_pctb"], "dd": last["dd_from_high"],
        "bb_upper": last["bb_upper"], "bb_lower": last["bb_lower"], "bb_mid": last["bb_mid"],
        "macd": last["MACD"], "macd_sig": last["MACD_sig"], "macd_hist": last["MACD_hist"],
        "atr": last["ATR_ratio"], "adx": last["ADX"],
        "plus_di": last["plus_di"], "minus_di": last["minus_di"],
        "high_20d": float(close.rolling(20).max().iloc[-1]),
        "ma200": last["MA200"],
        "vix_now": vix_now, "spy_now": spy_now, "spy_ma200_now": spy_ma200_now,
        "spy_regime": spy_regime,
        # 조건값
        "rsi_zone": cur_rsi_zone,
        "trend": "추세상승" if last["MA10"] > last["MA30"] else "추세하락",
        "macd_dir": "MACD상승" if last["MACD"] > last["MACD_sig"] else "MACD하락",
        "bb_zone": str(bb), "dd_zone": str(dd),
        "atr_zone": str(atr), "adx_zone": str(adx),
        "long_trend": "장기상승" if price > last["MA200"] else "장기하락",
        "market_regime": regime,
        "divergence": detect_divergence(close, data["RSI_W"]),
    }


def search_combinations(data, close, ctx, threshold_pct, min_n, crit, progress_cb=None):
    """9개 조건 전 조합 탐색.

    원본은 ~30,000개 조합마다 전체 배열에 마스크를 씌웠지만,
    여기선 groupby로 '실제 존재하는 조합'만 순회한다. 결과는 동일.
    """
    threshold = threshold_pct / 100
    n = len(data)
    close_arr = close.values

    ret_matrix = {}
    for h in HOLD_RANGE:
        arr = np.full(n, np.nan)
        if n > h:
            arr[:n - h] = close_arr[h:] / close_arr[:n - h] - 1
        ret_matrix[h] = arr

    keys_df = pd.DataFrame({
        "RSI구간": data["RSI_zone"].values, "추세": data["trend"].values,
        "BB위치": data["bb_zone"].values, "낙폭구간": data["dd_zone"].values,
        "MACD방향": data["macd_dir"].values, "ATR변동성": data["atr_zone"].values,
        "ADX추세강도": data["adx_zone"].values, "장기추세": data["long_trend"].values,
        "시장레짐": data["market_regime"].values,
    })
    groups = keys_df.groupby(COND_COLS, sort=False).indices  # {key_tuple: idx_array}

    rsi_num = data["RSI_W"].values
    magap_num = data["MA_gap_pct"].values
    bbpctb_num = data["bb_pctb"].values
    dd_num = data["dd_from_high"].values
    atr_num = data["ATR_ratio"].values
    adx_num = data["ADX"].values

    results = []
    total = len(groups)
    for gi, (key, idx) in enumerate(groups.items()):
        if progress_cb and gi % 200 == 0:
            progress_cb(gi / max(total, 1))
        if len(idx) < min_n:
            continue

        candidates = []
        for h in HOLD_RANGE:
            r = ret_matrix[h][idx]
            r = r[~np.isnan(r)]
            if len(r) < min_n:
                continue
            wr = (r > 0).mean()
            if wr < threshold:
                continue
            n_win = int((r > 0).sum())
            mean_ret, std_ret = r.mean(), r.std()
            loss_mean = r[r <= 0].mean() if (r <= 0).any() else np.nan
            rs = pd.Series(r)
            ev = calc_ev(rs)
            candidates.append({
                "보유일": h, "횟수": len(r),
                "승률%": round(wr * 100, 2),
                "평균수익%": round(mean_ret * 100, 2),
                "중앙값수익%": round(float(np.median(r)) * 100, 2),
                "손실평균%": round(loss_mean * 100, 2) if not np.isnan(loss_mean) else np.nan,
                "Sharpe": round(mean_ret / std_ret, 3) if std_ret > 0 else np.nan,
                "MDD%": round(calc_mdd(rs), 2),
                "기대값점수": ev if (ev is not None and not pd.isna(ev)) else -999,
                "유의성": sig_label(n_win, len(r)),
            })
        if not candidates:
            continue

        df_c = pd.DataFrame(candidates)
        top_sig = df_c[df_c["유의성"] == "★★ p<0.01"]
        pool = top_sig if len(top_sig) > 0 else df_c
        best = pool.loc[pool[crit].idxmax()]

        row = dict(zip(COND_COLS, key))
        row.update({
            "최적보유일": int(best["보유일"]), "횟수": int(best["횟수"]),
            "승률%": best["승률%"], "평균수익%": best["평균수익%"],
            "중앙값수익%": best["중앙값수익%"], "손실평균%": best["손실평균%"],
            "Sharpe": best["Sharpe"], "MDD%": best["MDD%"],
            "기대값점수": best["기대값점수"], "유의성": best["유의성"],
            "평균RSI": round(rsi_num[idx].mean(), 2),
            "평균MA이격%": round(magap_num[idx].mean(), 2),
            "평균BB%B": round(bbpctb_num[idx].mean(), 3),
            "평균낙폭%": round(dd_num[idx].mean(), 2),
            "평균ATR%": round(atr_num[idx].mean(), 2),
            "평균ADX": round(adx_num[idx].mean(), 2),
            "유효보유일범위": (f"{int(df_c['보유일'].min())}~{int(df_c['보유일'].max())}일"
                          if len(df_c) > 1 else f"{int(best['보유일'])}일"),
            "_cond_key": tuple(key),
        })
        results.append(row)

    if progress_cb:
        progress_cb(1.0)
    if not results:
        return None

    df = pd.DataFrame(results)
    df["기대값점수"] = df["기대값점수"].replace(-999, np.nan)
    return df


def assign_grades(df, crit, n_display):
    """🏆황금 / ★★신뢰 / 💰수익 등급 부여"""
    key7 = ["RSI구간", "추세", "BB위치", "낙폭구간", "MACD방향", "ATR변동성", "ADX추세강도"]
    top_keys = set(
        df.sort_values(crit, ascending=False).head(n_display)[key7].apply(tuple, axis=1)
    )
    is_sig = df["유의성"] == "★★ p<0.01"
    is_top = df[key7].apply(tuple, axis=1).isin(top_keys)

    def grade(s, t):
        if s and t: return "🏆 황금"
        if s: return "★★ 신뢰"
        if t: return "💰 수익"
        return ""

    df["🏆등급"] = [grade(s, t) for s, t in zip(is_sig, is_top)]
    return df


def count_match(df, cur):
    tgt = [cur["rsi_zone"], cur["trend"], cur["bb_zone"], cur["dd_zone"],
           cur["macd_dir"], cur["atr_zone"], cur["adx_zone"],
           cur["long_trend"], cur["market_regime"]]
    m = np.zeros(len(df), dtype=int)
    for col, t in zip(COND_COLS, tgt):
        m += (df[col].values == t).astype(int)
    return m


def build_cond_mask(data, key):
    rz, tr, bb, dd, md, az, adz, lt, mr = key
    return ((data["RSI_zone"] == rz) & (data["trend"] == tr) &
            (data["bb_zone"] == bb) & (data["dd_zone"] == dd) &
            (data["macd_dir"] == md) & (data["atr_zone"] == az) &
            (data["adx_zone"] == adz) & (data["long_trend"] == lt) &
            (data["market_regime"] == mr))


def current_full_mask(data, cur):
    key = (cur["rsi_zone"], cur["trend"], cur["bb_zone"], cur["dd_zone"],
           cur["macd_dir"], cur["atr_zone"], cur["adx_zone"],
           cur["long_trend"], cur["market_regime"])
    return build_cond_mask(data, key)
