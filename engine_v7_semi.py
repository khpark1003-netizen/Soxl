# -*- coding: utf-8 -*-
"""분석 엔진 v7 (반도체 특화판) — 원본 engine.py(v6)는 그대로 두고 새로 만든 파일

v6 대비 변경점
  1. BB위치 조건   → 거래량 비율 조건 (오늘 거래량 ÷ 20일 평균)
  2. 시장레짐      → SPY 200일선 대신 '반도체 상대강도(SOXX vs QQQ, 20일)' + VIX
                     (v6의 혼조장세는 논리상 나올 수 없어서, 3단계가 모두 나오도록 정리)
  3. 상관 진단     → SOXX 와 SPY·QQQ 의 60일 롤링 상관 (조건에는 미포함, 참고용)

조건 개수는 v6 과 같은 9개.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from indicators import (
    rsi_wilders, rsi_cutler, calc_macd, calc_atr, calc_adx,
    detect_divergence, calc_mdd, calc_ev, sig_label, get_data,
)

# ------------------------------------------------------------------
# 조정 가능한 값 (처음 잡은 시작값이며 실제 데이터로 튜닝한 값이 아님)
# ------------------------------------------------------------------
VOL_WINDOW = 20                 # 거래량 평균 기간
VOL_LOW, VOL_HIGH = 0.8, 1.5    # 거래량 비율 구간 경계 (평균 대비 배수)
RS_WINDOW = 20                  # 반도체 상대강도 계산 기간(거래일)
RS_WEAK = -3.0                  # 상대강도가 이 값(%) 미만이면 '반도체약세'
VIX_DANGER = 30.0               # VIX 가 이 값 이상이면 '위험장세'
CORR_WINDOW = 60                # 상관 계산 기간(거래일)

VOL_LABELS = ["거래량낮음", "거래량보통", "거래량급증"]
DD_LABELS = ["급락(-15%↓)", "큰낙폭(-15~-7%)", "중낙폭(-7~-3%)", "소낙폭(-3~-1%)", "고점근처"]
ATR_LABELS = ["낮음", "보통", "높음"]
ADX_LABELS = ["횡보", "약한추세", "강한추세"]
REGIME_LABELS = ["안정장세", "반도체약세", "위험장세"]
HOLD_RANGE = range(3, 21)
SHOW_HOLDS = [3, 5, 7, 10, 12, 15, 18, 20]
COND_COLS = ["RSI구간", "추세", "거래량", "낙폭구간", "MACD방향",
             "ATR변동성", "ADX추세강도", "장기추세", "시장레짐"]


# ------------------------------------------------------------------
# 시장 레짐 (VIX + 반도체 상대강도)
# ------------------------------------------------------------------
def classify_regime(vix, rs):
    """위험장세: VIX>=30 (시장 전체 공포)
       반도체약세: VIX<30 이면서 상대강도 < -3% (섹터만 약함)
       안정장세: 그 외"""
    out = pd.Series(
        np.where(vix >= VIX_DANGER, "위험장세",
                 np.where(rs < RS_WEAK, "반도체약세", "안정장세")),
        index=vix.index, dtype=object)
    return out.mask(vix.isna() | rs.isna(), np.nan)


def _must_get(ticker, start, what):
    """필수 데이터: 실패하면 조용히 넘어가지 않고 원인을 알려준다"""
    try:
        df = get_data(ticker, start)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{what}({ticker}) 데이터 로드 실패: {e}") from e
    if df is None or df.empty:
        raise RuntimeError(f"{what}({ticker}) 데이터가 비어 있습니다")
    return df


# ------------------------------------------------------------------
# 상관 진단
# ------------------------------------------------------------------
def build_corr(soxx, qqq, spy):
    rets = pd.DataFrame({"SOXX": soxx["Close"].pct_change(),
                         "QQQ": qqq["Close"].pct_change()})
    if spy is not None and len(spy) > 0:
        rets["SPY"] = spy["Close"].pct_change()
    rets = rets.dropna()
    out = pd.DataFrame(index=rets.index)
    out["SOXX-QQQ"] = rets["SOXX"].rolling(CORR_WINDOW).corr(rets["QQQ"])
    if "SPY" in rets:
        out["SOXX-SPY"] = rets["SOXX"].rolling(CORR_WINDOW).corr(rets["SPY"])
    return out.dropna()


def corr_summary(corr_df):
    out = {}
    if corr_df is None:
        return out
    for c in corr_df.columns:
        s = corr_df[c].dropna()
        if len(s) == 0:
            continue
        cur = float(s.iloc[-1])
        out[c] = {"current": cur, "mean_all": float(s.mean()),
                  "mean_1y": float(s.iloc[-252:].mean()),
                  "pctile": float((s <= cur).mean() * 100)}
    return out


# ------------------------------------------------------------------
# 로드 + 지표 계산
# ------------------------------------------------------------------
def load_and_build(years, ticker="SOXL"):
    start = datetime.today() - timedelta(days=365 * years)
    data = _must_get(ticker, start, "매매 대상")
    vix = _must_get("^VIX", start, "VIX")
    soxx = _must_get("SOXX", start, "반도체 ETF")
    qqq = _must_get("QQQ", start, "나스닥100 ETF")
    try:
        spy = get_data("SPY", start)      # 상관 진단 전용 (없어도 앱은 동작)
    except Exception:  # noqa: BLE001
        spy = None

    close, high, low = data["Close"], data["High"], data["Low"]
    data = data.copy()

    data["RSI_W"] = rsi_wilders(close)
    data["Sig_W"] = data["RSI_W"].rolling(6).mean()
    data["RSI_C"] = rsi_cutler(close)
    data["Sig_C"] = data["RSI_C"].rolling(6).mean()
    data["MA10"] = close.rolling(10).mean()
    data["MA30"] = close.rolling(30).mean()
    data["MA_gap_pct"] = (data["MA10"] - data["MA30"]) / data["MA30"] * 100
    data["dd_from_high"] = (close / close.rolling(20).max() - 1) * 100

    macd_line, macd_sig, macd_hist = calc_macd(close)
    data["MACD"], data["MACD_sig"], data["MACD_hist"] = macd_line, macd_sig, macd_hist
    data["ATR_ratio"] = calc_atr(high, low, close) / close * 100

    adx_val, plus_di, minus_di = calc_adx(high, low, close)
    data["ADX"], data["plus_di"], data["minus_di"] = adx_val, plus_di, minus_di

    # [신규] 거래량 비율
    vol = data["Volume"].astype(float)
    data["vol_avg"] = vol.rolling(VOL_WINDOW).mean()
    data["vol_ratio"] = (vol / data["vol_avg"]).replace([np.inf, -np.inf], np.nan)

    # 장기추세 (SOXL 200일선)
    data["MA200"] = close.rolling(200).mean()
    lt = pd.Series(np.where(close > data["MA200"], "장기상승", "장기하락"),
                   index=data.index, dtype=object)
    data["long_trend"] = lt.mask(data["MA200"].isna(), np.nan)

    # [변경] 시장 레짐: VIX + 반도체 상대강도(SOXX/QQQ 비율의 20일 변화, %)
    ratio = soxx["Close"] / qqq["Close"]
    rs = (ratio / ratio.shift(RS_WINDOW) - 1) * 100
    data["semi_rs"] = rs.reindex(data.index).ffill()
    data["vix_close"] = vix["Close"].reindex(data.index).ffill()
    data["market_regime"] = classify_regime(data["vix_close"], data["semi_rs"])

    data = data.dropna(subset=[
        "RSI_W", "Sig_W", "RSI_C", "Sig_C", "MA10", "MA30", "dd_from_high",
        "MACD", "MACD_sig", "ATR_ratio", "ADX", "MA200", "long_trend",
        "vol_ratio", "semi_rs", "vix_close", "market_regime",
    ])
    if len(data) < 300:
        raise RuntimeError(f"지표 계산 후 남은 데이터가 너무 적습니다({len(data)}행)")

    # RSI 구간 (분위수 기반)
    pcts = [0, 15, 30, 45, 55, 70, 85, 100]
    rsi_edges = np.unique(np.round(np.percentile(data["RSI_W"].values, pcts), 1))
    rsi_labels = [f"RSI {rsi_edges[i]:.0f}~{rsi_edges[i+1]:.0f}"
                  for i in range(len(rsi_edges) - 1)]

    data["RSI_zone"] = pd.cut(data["RSI_W"], bins=rsi_edges, labels=rsi_labels,
                              include_lowest=True).astype(str)
    data["vol_zone"] = pd.cut(data["vol_ratio"], bins=[-999, VOL_LOW, VOL_HIGH, 999],
                              labels=VOL_LABELS).astype(str)
    data["dd_zone"] = pd.cut(data["dd_from_high"], bins=[-999, -15, -7, -3, -1, .1],
                             labels=DD_LABELS).astype(str)
    data["atr_zone"] = pd.cut(data["ATR_ratio"], bins=[-999, 3, 6, 999],
                              labels=ATR_LABELS).astype(str)
    data["adx_zone"] = pd.cut(data["ADX"], bins=[-999, 20, 40, 999],
                              labels=ADX_LABELS).astype(str)
    data["trend"] = np.where(data["MA10"] > data["MA30"], "추세상승", "추세하락")
    data["macd_dir"] = np.where(data["MACD"] > data["MACD_sig"], "MACD상승", "MACD하락")

    close = close.reindex(data.index)
    ctx = {"rsi_labels": rsi_labels, "rsi_edges": rsi_edges,
           "corr_df": build_corr(soxx, qqq, spy)}
    return data, close, ctx


# ------------------------------------------------------------------
# 현재 상태
# ------------------------------------------------------------------
def current_state(data, close, ctx):
    last = data.iloc[-1]
    price = float(close.iloc[-1])

    cur_rsi_zone = None
    edges, labels = ctx["rsi_edges"], ctx["rsi_labels"]
    for i in range(len(edges) - 1):
        if edges[i] <= last["RSI_W"] <= edges[i + 1]:
            cur_rsi_zone = labels[i]
            break

    vol_z = pd.cut([last["vol_ratio"]], bins=[-999, VOL_LOW, VOL_HIGH, 999], labels=VOL_LABELS)[0]
    dd = pd.cut([last["dd_from_high"]], bins=[-999, -15, -7, -3, -1, .1], labels=DD_LABELS)[0]
    atr = pd.cut([last["ATR_ratio"]], bins=[-999, 3, 6, 999], labels=ATR_LABELS)[0]
    adx = pd.cut([last["ADX"]], bins=[-999, 20, 40, 999], labels=ADX_LABELS)[0]
    vix = float(last["vix_close"])
    vix_zone = str(pd.cut([vix], bins=[-999, 20, 30, 40, 999],
                          labels=["안정", "경계", "공포", "패닉"])[0])

    return {
        "price": price,
        "rsi_w": last["RSI_W"], "sig_w": last["Sig_W"],
        "rsi_c": last["RSI_C"], "sig_c": last["Sig_C"],
        "ma10": last["MA10"], "ma30": last["MA30"], "magap": last["MA_gap_pct"],
        "macd": last["MACD"], "macd_sig": last["MACD_sig"], "macd_hist": last["MACD_hist"],
        "atr": last["ATR_ratio"], "adx": last["ADX"],
        "plus_di": last["plus_di"], "minus_di": last["minus_di"],
        "dd": last["dd_from_high"],
        "high_20d": float(close.rolling(20).max().iloc[-1]),
        "ma200": last["MA200"],
        "vol_ratio": float(last["vol_ratio"]),
        "vol_today": float(data["Volume"].iloc[-1]), "vol_avg": float(last["vol_avg"]),
        "vix": vix, "vix_zone": vix_zone, "semi_rs": float(last["semi_rs"]),
        # 조건값
        "rsi_zone": cur_rsi_zone,
        "trend": "추세상승" if last["MA10"] > last["MA30"] else "추세하락",
        "macd_dir": "MACD상승" if last["MACD"] > last["MACD_sig"] else "MACD하락",
        "vol_zone": str(vol_z), "dd_zone": str(dd),
        "atr_zone": str(atr), "adx_zone": str(adx),
        "long_trend": "장기상승" if price > last["MA200"] else "장기하락",
        "market_regime": str(last["market_regime"]),
        "divergence": detect_divergence(close, data["RSI_W"]),
    }


# ------------------------------------------------------------------
# 조합 탐색 (groupby 로 실제 존재하는 조합만 순회)
# ------------------------------------------------------------------
def search_combinations(data, close, ctx, threshold_pct, min_n, crit, progress_cb=None):
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
        "거래량": data["vol_zone"].values, "낙폭구간": data["dd_zone"].values,
        "MACD방향": data["macd_dir"].values, "ATR변동성": data["atr_zone"].values,
        "ADX추세강도": data["adx_zone"].values, "장기추세": data["long_trend"].values,
        "시장레짐": data["market_regime"].values,
    })
    groups = keys_df.groupby(COND_COLS, sort=False).indices

    rsi_num = data["RSI_W"].values
    magap_num = data["MA_gap_pct"].values
    vol_num = data["vol_ratio"].values
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
            rs_ = pd.Series(r)
            ev = calc_ev(rs_)
            candidates.append({
                "보유일": h, "횟수": len(r),
                "승률%": round(wr * 100, 2),
                "평균수익%": round(mean_ret * 100, 2),
                "중앙값수익%": round(float(np.median(r)) * 100, 2),
                "손실평균%": round(loss_mean * 100, 2) if not np.isnan(loss_mean) else np.nan,
                "Sharpe": round(mean_ret / std_ret, 3) if std_ret > 0 else np.nan,
                "MDD%": round(calc_mdd(rs_), 2),
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
            "평균거래량비": round(vol_num[idx].mean(), 2),
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
    key7 = ["RSI구간", "추세", "거래량", "낙폭구간", "MACD방향", "ATR변동성", "ADX추세강도"]
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


def _cur_key(cur):
    return (cur["rsi_zone"], cur["trend"], cur["vol_zone"], cur["dd_zone"],
            cur["macd_dir"], cur["atr_zone"], cur["adx_zone"],
            cur["long_trend"], cur["market_regime"])


def count_match(df, cur):
    m = np.zeros(len(df), dtype=int)
    for col, t in zip(COND_COLS, _cur_key(cur)):
        m += (df[col].values == t).astype(int)
    return m


def build_cond_mask(data, key):
    rz, tr, vz, dd, md, az, adz, lt, mr = key
    return ((data["RSI_zone"] == rz) & (data["trend"] == tr) &
            (data["vol_zone"] == vz) & (data["dd_zone"] == dd) &
            (data["macd_dir"] == md) & (data["atr_zone"] == az) &
            (data["adx_zone"] == adz) & (data["long_trend"] == lt) &
            (data["market_regime"] == mr))


def current_full_mask(data, cur):
    return build_cond_mask(data, _cur_key(cur))
