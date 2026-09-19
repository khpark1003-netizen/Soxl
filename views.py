# -*- coding: utf-8 -*-
"""HTML 렌더링 함수 모음 (display(HTML) -> 문자열 반환)
Streamlit에서는 st.markdown(html, unsafe_allow_html=True) 로 출력한다."""

import numpy as np
import pandas as pd

from indicators import calc_mdd, calc_ev, sig_label
from engine import SHOW_HOLDS

WRAP = "font-family:monospace;background:#1c1c1c;border-radius:10px;padding:18px;color:#eee;margin-top:12px;"


def _wc(v): return "#00c853" if v >= 75 else "#ffab00" if v >= 60 else "#ef5350"
def _ec(v):
    if v is None or pd.isna(v): return "#aaa"
    return "#00c853" if v >= 7 else "#ffab00" if v >= 3 else "#ef5350"


def _scroll(inner):
    """모바일에서 표가 잘리지 않도록 가로 스크롤 래퍼"""
    return f"<div style='overflow-x:auto;-webkit-overflow-scrolling:touch'>{inner}</div>"


# ------------------------------------------------------------------
def holding_analysis_html(data, close, cond_mask, title):
    cm = cond_mask.reindex(data.index).fillna(False)
    rows = []
    for h in SHOW_HOLDS:
        r = (close.shift(-h) / close - 1).reindex(data.index)[cm].dropna()
        if len(r) < 5:
            continue
        rows.append({
            "보유일": h, "샘플": len(r),
            "승률%": round((r > 0).mean() * 100, 1),
            "평균수익%": round(r.mean() * 100, 2),
            "중앙값%": round(r.median() * 100, 2),
            "손실평균%": round(r[r <= 0].mean() * 100, 2) if (r <= 0).any() else np.nan,
            "기대값": calc_ev(r),
            "MDD%": round(calc_mdd(r), 2),
            "유의성": sig_label(int((r > 0).sum()), len(r)),
        })
    if not rows:
        return ""
    df_h = pd.DataFrame(rows)
    best_idx = df_h["기대값"].dropna().idxmax() if not df_h["기대값"].isna().all() else None

    body = ""
    for i, row in df_h.iterrows():
        is_best = (i == best_idx)
        bg = "background:#1a3a1a;" if is_best else ""
        tag = " ★ 최적" if is_best else ""
        ev_val = round(row["기대값"], 2) if not pd.isna(row["기대값"]) else "-"
        lm_val = round(row["손실평균%"], 2) if not pd.isna(row["손실평균%"]) else "-"
        body += f"""
        <tr style='border-bottom:1px solid #2a2a2a;{bg}'>
          <td style='padding:7px 10px;color:{"#ffca28" if is_best else "#ccc"};font-weight:{"bold" if is_best else "normal"};white-space:nowrap'>{int(row["보유일"])}일{tag}</td>
          <td style='padding:7px 10px;color:#aaa;text-align:right'>{int(row["샘플"])}회</td>
          <td style='padding:7px 10px;font-weight:bold;color:{_wc(row["승률%"])};text-align:right'>{row["승률%"]}%</td>
          <td style='padding:7px 10px;color:#fff;text-align:right'>{row["평균수익%"]}%</td>
          <td style='padding:7px 10px;color:#aaa;text-align:right'>{row["중앙값%"]}%</td>
          <td style='padding:7px 10px;color:#ef9a9a;text-align:right'>{lm_val}%</td>
          <td style='padding:7px 10px;font-weight:bold;color:{_ec(row["기대값"])};text-align:right'>{ev_val}</td>
          <td style='padding:7px 10px;color:#ef5350;text-align:right'>{row["MDD%"]}%</td>
          <td style='padding:7px 10px;color:#90caf9;text-align:right;font-size:11px;white-space:nowrap'>{row["유의성"]}</td>
        </tr>"""

    heads = ["보유일", "샘플", "승률%", "평균수익%", "중앙값%", "손실평균%", "기대값", "MDD%", "유의성"]
    head_html = "".join(
        f"<td style='padding:5px 10px;color:#555;{'' if i == 0 else 'text-align:right'}'>{h}</td>"
        for i, h in enumerate(heads))
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:15px;font-weight:bold;color:#00e676;margin-bottom:12px;border-bottom:1px solid #333;padding-bottom:8px'>💰 {title}</div>
      {_scroll(f"<table style='width:100%;border-collapse:collapse;font-size:13px'><tr style='border-bottom:1px solid #333'>{head_html}</tr>{body}</table>")}
      <div style='margin-top:10px;font-size:11px;color:#555;border-top:1px solid #333;padding-top:8px'>
        ★ 최적 = 기대값 기준 최적 보유일 &nbsp;|&nbsp; 손실평균% = 손실 거래만의 평균 손실폭 &nbsp;|&nbsp; MDD% = 해당 보유일 기준 최대 낙폭
      </div>
    </div>"""


# ------------------------------------------------------------------
def sample_details_html(data, close, cond_mask, hold_days=None,
                        title="조건 일치 개별 샘플 상세", max_samples=30):
    cm = cond_mask.reindex(data.index).fillna(False)
    match_dates = data.index[cm]
    if len(match_dates) == 0:
        return ""

    day_cols = [1, 2, 3, 5, 7, 10, 12, 15, 18, 20]
    if hold_days and hold_days not in day_cols:
        day_cols = sorted(set(day_cols + [int(hold_days)]))

    pos_of = {d: i for i, d in enumerate(data.index)}
    n = len(data)
    close_r = close.reindex(data.index)

    samples = []
    for dt in match_dates:
        i = pos_of[dt]
        entry = float(close_r.iloc[i])
        rets = {}
        for d in day_cols:
            j = i + d
            rets[d] = round((float(close_r.iloc[j]) / entry - 1) * 100, 2) if j < n else None
        final_h = hold_days if hold_days else day_cols[-1]
        samples.append({
            "날짜": dt.strftime("%Y-%m-%d"), "진입가": round(entry, 2),
            "RSI": round(data["RSI_W"].iloc[i], 1),
            "MA이격%": round(data["MA_gap_pct"].iloc[i], 2),
            "BB%B": round(data["bb_pctb"].iloc[i], 2),
            "낙폭%": round(data["dd_from_high"].iloc[i], 2),
            "ATR%": round(data["ATR_ratio"].iloc[i], 2),
            "ADX": round(data["ADX"].iloc[i], 1),
            "장기추세": data["long_trend"].iloc[i],
            "시장레짐": data["market_regime"].iloc[i],
            "_rets": rets, "_final": rets.get(final_h),
        })

    samples = samples[::-1]
    total_n, shown = len(samples), samples[:max_samples]
    win_n = sum(1 for s in samples if s["_final"] is not None and s["_final"] > 0)
    valid_n = sum(1 for s in samples if s["_final"] is not None)

    def cell(v):
        if v is None: return "#555", "-"
        c = "#00c853" if v >= 5 else "#66bb6a" if v > 0 else "#ef9a9a" if v > -5 else "#ef5350"
        return c, f"{v:+.2f}%"

    header_days = "".join(f"<td style='padding:5px 8px;color:#555;text-align:right'>D+{d}</td>" for d in day_cols)
    rows_html = ""
    for s in shown:
        cells = ""
        for d in day_cols:
            c, txt = cell(s["_rets"][d])
            is_final = bool(hold_days and d == hold_days)
            cells += (f"<td style='padding:5px 8px;text-align:right;"
                      f"{'font-weight:bold;background:#1a2a1a;' if is_final else ''}color:{c}'>{txt}</td>")
        lt_c = "#66bb6a" if s["장기추세"] == "장기상승" else "#ef5350"
        mr_c = {"안정장세": "#00c853", "혼조장세": "#ffab00"}.get(s["시장레짐"], "#ef5350")
        rows_html += f"""
        <tr style='border-bottom:1px solid #2a2a2a'>
          <td style='padding:5px 10px;color:#ffca28;white-space:nowrap'>{s["날짜"]}</td>
          <td style='padding:5px 8px;color:#fff;text-align:right'>{s["진입가"]}</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["RSI"]}</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["MA이격%"]}%</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["BB%B"]}</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["낙폭%"]}%</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["ATR%"]}%</td>
          <td style='padding:5px 8px;color:#aaa;text-align:right'>{s["ADX"]}</td>
          <td style='padding:5px 8px;text-align:right;color:{lt_c}'>{s["장기추세"]}</td>
          <td style='padding:5px 8px;text-align:right;color:{mr_c}'>{s["시장레짐"]}</td>
          {cells}
        </tr>"""

    note = "" if total_n <= max_samples else (
        f"<span style='color:#888;font-weight:normal'> (최신 {max_samples}개만 표시, 전체 {total_n}개)</span>")
    sub = ""
    if hold_days:
        wr = round(win_n / valid_n * 100, 1) if valid_n else 0
        sub = f"최종({hold_days}일) 기준 {win_n}/{valid_n}승 ({wr}%) — 강조된 D+{hold_days} 열이 최적보유일 기준 결과값"
    heads = ["날짜", "진입가", "RSI", "MA이격", "BB%B", "낙폭", "ATR", "ADX", "장기추세", "시장레짐"]
    head_html = "".join(
        f"<td style='padding:5px 8px;color:#555;{'' if i == 0 else 'text-align:right'}'>{h}</td>"
        for i, h in enumerate(heads))
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:15px;font-weight:bold;color:#ce93d8;margin-bottom:4px;border-bottom:1px solid #333;padding-bottom:8px'>🔬 {title} — 총 {total_n}개{note}</div>
      <div style='font-size:11px;color:#888;margin-bottom:10px'>{sub}</div>
      {_scroll(f"<table style='border-collapse:collapse;font-size:12px;white-space:nowrap'><tr style='border-bottom:1px solid #333'>{head_html}{header_days}</tr>{rows_html}</table>")}
      <div style='margin-top:10px;font-size:11px;color:#555;border-top:1px solid #333;padding-top:8px'>
        D+n = 진입 후 n거래일째 누적 수익률 &nbsp;|&nbsp; 최신 날짜부터 표시
      </div>
    </div>"""


# ------------------------------------------------------------------
def yearly_stats_html(data, close, cond_mask, hold_days, title="연도별 승률"):
    d = data.copy()
    d["_year"] = d.index.year
    cm = cond_mask.reindex(data.index).fillna(False)
    rows = []
    for yr in sorted(d["_year"].unique()):
        idx = d[d["_year"] == yr].index
        cm_y = cm.reindex(idx).fillna(False)
        r = (close.shift(-hold_days) / close - 1).reindex(idx)[cm_y].dropna()
        if len(r) < 3:
            continue
        rows.append({"연도": yr, "샘플": len(r), "승률%": round((r > 0).mean() * 100, 1),
                     "평균수익%": round(r.mean() * 100, 2), "기대값": calc_ev(r)})
    if not rows:
        return ""
    df_yr = pd.DataFrame(rows)

    body = ""
    for _, row in df_yr.iterrows():
        ev = "-" if pd.isna(row["기대값"]) else round(row["기대값"], 2)
        body += f"""
        <tr style='border-bottom:1px solid #2a2a2a'>
          <td style='padding:6px 10px;color:#ccc'>{int(row["연도"])}</td>
          <td style='padding:6px 10px;color:#aaa;text-align:right'>{int(row["샘플"])}회</td>
          <td style='padding:6px 10px;font-weight:bold;color:{_wc(row["승률%"])};text-align:right'>{row["승률%"]}%</td>
          <td style='padding:6px 10px;color:#fff;text-align:right'>{row["평균수익%"]}%</td>
          <td style='padding:6px 10px;font-weight:bold;color:{_ec(row["기대값"])};text-align:right'>{ev}</td>
        </tr>"""

    total_yr = len(df_yr)
    above = int((df_yr["승률%"] >= 60).sum())
    consist = round(above / total_yr * 100) if total_yr else 0
    cc = "#00c853" if consist >= 80 else "#ffab00" if consist >= 60 else "#ef5350"
    cl = "매우 일관됨" if consist >= 80 else "보통" if consist >= 60 else "불안정 — 특정 연도 편중 주의"
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:14px;font-weight:bold;color:#90caf9;margin-bottom:12px;border-bottom:1px solid #333;padding-bottom:8px'>📅 {title}</div>
      {_scroll(f"<table style='width:100%;border-collapse:collapse;font-size:13px'><tr style='border-bottom:1px solid #333'><td style='padding:5px 10px;color:#555'>연도</td><td style='padding:5px 10px;color:#555;text-align:right'>샘플</td><td style='padding:5px 10px;color:#555;text-align:right'>승률%</td><td style='padding:5px 10px;color:#555;text-align:right'>평균수익%</td><td style='padding:5px 10px;color:#555;text-align:right'>기대값</td></tr>{body}</table>")}
      <div style='margin-top:10px;font-size:12px;border-top:1px solid #333;padding-top:8px'>
        일관성: <b style='color:{cc}'>{consist}% ({above}/{total_yr}년 승률 60%↑)</b> → <span style='color:{cc}'>{cl}</span>
      </div>
    </div>"""


# ------------------------------------------------------------------
def period_comparison_html(data, close, cond_mask, hold_days):
    cutoff = data.index.max() - pd.DateOffset(years=2)
    cm = cond_mask.reindex(data.index).fillna(False)

    def stats(idx_filter):
        cm_f = cm.reindex(idx_filter).fillna(False)
        r = (close.shift(-hold_days) / close - 1).reindex(idx_filter)[cm_f].dropna()
        if len(r) < 5:
            return None
        return {"샘플": len(r), "승률%": round((r > 0).mean() * 100, 1),
                "평균수익%": round(r.mean() * 100, 2), "중앙값%": round(r.median() * 100, 2),
                "기대값": calc_ev(r), "MDD%": round(calc_mdd(r), 2)}

    s_all = stats(data.index)
    s_rec = stats(data[data.index >= cutoff].index)
    if not s_all or not s_rec:
        return ""

    def fmt(v): return "-" if v is None or pd.isna(v) else str(round(v, 2))
    def dc(a, b): return "#aaa" if pd.isna(a) or pd.isna(b) else "#00c853" if b >= a else "#ef5350"
    def ds(a, b):
        if pd.isna(a) or pd.isna(b): return ""
        d = round(b - a, 2)
        return f"({'▲' if d > 0 else '▼'}{abs(d)})"

    metrics = [("샘플수", s_all["샘플"], s_rec["샘플"], False),
               ("승률%", s_all["승률%"], s_rec["승률%"], True),
               ("평균수익%", s_all["평균수익%"], s_rec["평균수익%"], True),
               ("중앙값%", s_all["중앙값%"], s_rec["중앙값%"], True),
               ("기대값", s_all["기대값"], s_rec["기대값"], True),
               ("MDD%", s_all["MDD%"], s_rec["MDD%"], False)]
    body = ""
    for label, va, vr, hb in metrics:
        c = dc(va, vr) if hb else "#aaa"
        d = ds(va, vr) if hb else ""
        body += f"""
        <tr style='border-bottom:1px solid #2a2a2a'>
          <td style='padding:6px 10px;color:#ccc'>{label}</td>
          <td style='padding:6px 10px;color:#aaa;text-align:right'>{fmt(va)}</td>
          <td style='padding:6px 10px;font-weight:bold;color:{c};text-align:right'>{fmt(vr)} <span style='font-size:11px'>{d}</span></td>
        </tr>"""

    r_wr, a_wr = s_rec["승률%"], s_all["승률%"]
    r_ev = s_rec["기대값"] if not pd.isna(s_rec["기대값"]) else 0
    a_ev = s_all["기대값"] if not pd.isna(s_all["기대값"]) else 0
    if r_wr >= a_wr and r_ev >= a_ev:
        jc, jt = "#00c853", "최근에도 유효 — 패턴 지속 중"
    elif r_wr >= a_wr * 0.9:
        jc, jt = "#ffab00", "소폭 약화 — 참고하되 주의"
    else:
        jc, jt = "#ef5350", "최근 성과 저하 — 과적합 가능성 검토 필요"
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:14px;font-weight:bold;color:#90caf9;margin-bottom:12px;border-bottom:1px solid #333;padding-bottom:8px'>🔍 전체 기간 vs 최근 2년 비교</div>
      {_scroll(f"<table style='width:100%;border-collapse:collapse;font-size:13px'><tr style='border-bottom:1px solid #333'><td style='padding:5px 10px;color:#555'>지표</td><td style='padding:5px 10px;color:#555;text-align:right'>전체</td><td style='padding:5px 10px;color:#555;text-align:right'>최근 2년</td></tr>{body}</table>")}
      <div style='margin-top:10px;font-size:12px;border-top:1px solid #333;padding-top:8px'>종합: <b style='color:{jc}'>{jt}</b></div>
    </div>"""


# ------------------------------------------------------------------
def glossary_html():
    def item(title, body, last=False):
        bd = "" if last else "border-bottom:1px solid #2a2a2a;padding-bottom:14px;"
        return (f"<div style='margin-bottom:14px;{bd}'>"
                f"<div style='color:#ffca28;font-weight:bold;margin-bottom:5px'>📌 {title}</div>"
                f"<div style='color:#ccc;font-size:12px;line-height:1.7'>{body}</div></div>")

    left = "".join([
        item("RSI (상대강도지수)",
             "<b style='color:#90caf9'>의미:</b> 상승폭/전체폭×100. 0~100.<br>"
             "<b style='color:#ef5350'>70↑ 과매수</b> / <b style='color:#66bb6a'>30↓ 과매도</b>.<br>"
             "Wilder's=rolling(미래에셋기준) / Cutler's=EWM.<br>Signal=RSI 6일MA. RSI&gt;Signal → 상승모멘텀."),
        item("MACD",
             "<b style='color:#90caf9'>의미:</b> 단기/장기 EMA 차이로 추세 전환 포착.<br>"
             "MACD선=12EMA-26EMA / Signal=MACD 9일EMA.<br>"
             "<b style='color:#66bb6a'>골든크로스:</b> MACD&gt;Signal → 상승.<br>"
             "<b style='color:#ef5350'>데드크로스:</b> MACD&lt;Signal → 하락.<br>RSI=\"과열도\" / MACD=\"방향\" — 서로 보완."),
        item("볼린저밴드 %B",
             "<b style='color:#90caf9'>의미:</b> 20일MA±2σ 밴드에서 현재가 위치.<br>"
             "%B=(현재가-하단)/(상단-하단).<br>1.0=상단 / 0.5=중간 / 0.0=하단.<br>BB하단(&lt;0.2)~BB상단(&gt;0.8)."),
        item("RSI 다이버전스",
             "<b style='color:#90caf9'>의미:</b> 가격과 RSI가 반대 방향. 추세 반전 조기 경고.<br>"
             "<b style='color:#66bb6a'>상승:</b> 가격↓ + RSI저점↑ → 반등 임박.<br>"
             "<b style='color:#ef5350'>하락:</b> 가격↑ + RSI고점↓ → 조정 임박.<br>단독 사용보다 다른 지표와 함께 확인 필요.", last=True),
    ])
    right = "".join([
        item("ATR 변동성",
             "<b style='color:#90caf9'>의미:</b> 14일 일일 가격변동폭 평균. 시장 흔들림 강도.<br>ATR비율=ATR/현재가×100 (% 단위).<br>"
             "<b style='color:#66bb6a'>낮음(&lt;3%):</b> 안정. 신호 신뢰도↑.<br>"
             "<b style='color:#ffab00'>보통(3~6%):</b> 일반적 변동성.<br>"
             "<b style='color:#ef5350'>높음(&gt;6%):</b> 급등락. 레버리지 진입 위험↑."),
        item("ADX 추세강도",
             "<b style='color:#90caf9'>의미:</b> 추세 강도를 0~100으로 표현. 방향이 아닌 강도만.<br>"
             "<b style='color:#ef5350'>횡보(&lt;20):</b> 추세없음. MACD신호 신뢰도↓.<br>"
             "<b style='color:#ffab00'>약한추세(20~40):</b> 추세 형성 중.<br>"
             "<b style='color:#66bb6a'>강한추세(&gt;40):</b> 강한추세. 추세추종 유리."),
        item("MA10/MA30 추세",
             "<b style='color:#66bb6a'>정배열(MA10&gt;MA30):</b> 상승추세.<br>"
             "<b style='color:#ef5350'>역배열(MA10&lt;MA30):</b> 하락추세.<br>이격% 클수록 추세 탄력 강함."),
        item("장기추세 (200일선)",
             "<b style='color:#90caf9'>의미:</b> 종가가 200일 이동평균 위/아래인지.<br>"
             "<b style='color:#66bb6a'>장기상승:</b> 구조적 상승장 안의 눌림목.<br>"
             "<b style='color:#ef5350'>장기하락:</b> 구조적 하락장 — 데드캣 바운스 위험."),
        item("시장 레짐 (SPY·VIX)",
             "<b style='color:#90caf9'>의미:</b> SOXL 개별 지표가 아닌 시장 전체 상태.<br>"
             "<b style='color:#00c853'>안정장세:</b> SPY상승장+VIX낮음.<br>"
             "<b style='color:#ffab00'>혼조장세:</b> 방향성 불분명.<br>"
             "<b style='color:#ef5350'>위험장세:</b> SPY하락장 또는 VIX공포/패닉."),
        item("유의성 / 기대값",
             "<b style='color:#90caf9'>유의성(이항검정):</b> 승률이 우연이 아님을 검증.<br>★★p&lt;0.01=우연확률1%미만.<br>"
             "<b style='color:#90caf9'>기대값:</b> 승률×수익-(1-승률)×손실.<br>"
             "<b style='color:#66bb6a'>7↑좋음</b> / <b style='color:#ffab00'>3~7보통</b> / <b style='color:#ef5350'>0↓손실구조</b>.<br>"
             "샘플 30개↑ + ★★ 동시 충족 시 신뢰.", last=True),
    ])
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:15px;font-weight:bold;color:#90caf9;margin-bottom:14px;border-bottom:1px solid #333;padding-bottom:8px'>📖 지표 용어 상세 설명</div>
      <div style='display:flex;flex-wrap:wrap;gap:0 28px'>
        <div style='flex:1 1 300px;min-width:0'>{left}</div>
        <div style='flex:1 1 300px;min-width:0'>{right}</div>
      </div>
    </div>"""


# ------------------------------------------------------------------
def market_state_html(c):
    def rc(v): return "#ef5350" if v >= 70 else "#66bb6a" if v <= 30 else "#fff"
    def vc(v): return "#aaa" if np.isnan(v) else "#66bb6a" if v < 20 else "#ffab00" if v < 30 else "#ef5350"
    mc = "#66bb6a" if c["macd"] > c["macd_sig"] else "#ef5350"
    stc = "#66bb6a" if c["ma10"] > c["ma30"] else "#ef5350"
    atrc = "#66bb6a" if c["atr"] < 3 else "#ffab00" if c["atr"] < 6 else "#ef5350"
    adxc = "#66bb6a" if c["adx"] > 40 else "#ffab00" if c["adx"] > 20 else "#ef5350"
    div = c["divergence"]
    divc = "#66bb6a" if div == "상승 다이버전스" else "#ef5350" if div == "하락 다이버전스" else "#aaa"
    ltc = "#66bb6a" if c["long_trend"] == "장기상승" else "#ef5350"
    mrc = {"안정장세": "#00c853", "혼조장세": "#ffab00", "위험장세": "#ef5350"}[c["market_regime"]]
    r2 = lambda v: "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, 2)
    ddc = "#ef5350" if c["dd"] < -7 else "#ffab00" if c["dd"] < -3 else "#fff"
    di_up = c["plus_di"] > c["minus_di"]

    return f"""
    <div style='{WRAP}'>
      <div style='font-size:15px;font-weight:bold;color:#90caf9;margin-bottom:14px;border-bottom:1px solid #333;padding-bottom:8px'>
        📊 현재 시장 상태 <span style='font-size:11px;font-weight:normal;color:#555'>※ 전략기준: Wilder's RSI (미래에셋 동일)</span>
      </div>
      <div style='display:flex;flex-wrap:wrap;gap:0 28px'>
        <div style='flex:1 1 300px;min-width:0'>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>RSI (14일)</div>
            <table style='font-size:13px;border-collapse:collapse'>
              <tr><td style='padding:3px 8px;color:#aaa'>Wilder's</td><td style='padding:3px 8px;font-weight:bold;color:{rc(c["rsi_w"])}'>{round(c["rsi_w"],2)}</td>
                  <td style='padding:3px 8px;color:#aaa'>Signal</td><td style='padding:3px 8px'>{round(c["sig_w"],2)}</td></tr>
              <tr><td style='padding:3px 8px;color:#aaa'>Cutler's</td><td style='padding:3px 8px;font-weight:bold;color:{rc(c["rsi_c"])}'>{round(c["rsi_c"],2)}</td>
                  <td style='padding:3px 8px;color:#aaa'>Signal</td><td style='padding:3px 8px'>{round(c["sig_c"],2)}</td></tr>
            </table></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>MACD (12/26/9)</div>
            <div style='font-size:13px;padding:2px 8px'>MACD <b>{round(c["macd"],3)}</b> Signal <b>{round(c["macd_sig"],3)}</b>
              Hist <b style='color:{mc}'>{round(c["macd_hist"],3)}</b> → <b style='color:{mc}'>{c["macd_dir"]} {"(골든크로스)" if c["macd"]>c["macd_sig"] else "(데드크로스)"}</b></div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>MA10 / MA30</div>
            <div style='font-size:13px;padding:2px 8px'>MA10 <b>{round(c["ma10"],2)}</b> MA30 <b>{round(c["ma30"],2)}</b>
              이격 <b style='color:{stc}'>{round(c["magap"],2)}%</b> → <b style='color:{stc}'>{c["trend"]}</b></div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>볼린저밴드</div>
            <div style='font-size:13px;padding:2px 8px'>상단 <b style='color:#ef5350'>{round(c["bb_upper"],2)}</b> 중간 <b>{round(c["bb_mid"],2)}</b> 하단 <b style='color:#66bb6a'>{round(c["bb_lower"],2)}</b><br>
              현재가 <b style='color:#ffca28'>{round(c["price"],2)}</b> %B <b>{round(c["bbpctb"],3)}</b> → <b>{c["bb_zone"]}</b></div></div>
        </div>
        <div style='flex:1 1 300px;min-width:0'>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>ATR 변동성</div>
            <div style='font-size:13px;padding:2px 8px'>ATR비율 <b style='color:{atrc}'>{round(c["atr"],2)}%</b> → <b style='color:{atrc}'>{c["atr_zone"]}</b>
              <span style='font-size:11px;color:#555'>(낮음&lt;3% / 보통3~6% / 높음&gt;6%)</span></div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>ADX 추세강도</div>
            <div style='font-size:13px;padding:2px 8px'>ADX <b style='color:{adxc}'>{round(c["adx"],2)}</b> → <b style='color:{adxc}'>{c["adx_zone"]}</b><br>
              +DI <b style='color:#66bb6a'>{round(c["plus_di"],2)}</b> -DI <b style='color:#ef5350'>{round(c["minus_di"],2)}</b>
              → <b style='color:{"#66bb6a" if di_up else "#ef5350"}'>{"+DI 우세(상승압력)" if di_up else "-DI 우세(하락압력)"}</b></div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>20일 고점 대비 낙폭</div>
            <div style='font-size:13px;padding:2px 8px'>고점 <b>{round(c["high_20d"],2)}</b> 낙폭 <b style='color:{ddc}'>{round(c["dd"],2)}%</b> → <b>{c["dd_zone"]}</b></div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>RSI 다이버전스 경고</div>
            <div style='font-size:13px;padding:2px 8px'><b style='color:{divc}'>{"⚠️ " if div else "✅ "}{div if div else "없음"}</b>
              {f"<span style='font-size:11px;color:#aaa'>— {'반등 임박' if div=='상승 다이버전스' else '조정 임박'}</span>" if div else ""}</div></div>
          <div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>장기추세 (SOXL 200일선)</div>
            <div style='font-size:13px;padding:2px 8px'>현재가 <b>{round(c["price"],2)}</b> vs MA200 <b>{round(c["ma200"],2)}</b> → <b style='color:{ltc}'>{c["long_trend"]}</b></div></div>
          <div style='border-top:1px solid #333;padding-top:10px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>🌐 시장 레짐 (SPY 200일선 + VIX)</div>
            <div style='font-size:13px;padding:2px 8px'>VIX <b style='color:{vc(c["vix_now"])}'>{r2(c["vix_now"])}</b> | SPY <b>{r2(c["spy_now"])}</b> vs MA200 <b>{r2(c["spy_ma200_now"])}</b>
              → <b style='color:{"#66bb6a" if c["spy_regime"]=="상승장" else "#ef5350"}'>{c["spy_regime"]}</b><br>
              종합 <b style='color:{mrc}'>{c["market_regime"]}</b>
              <span style='font-size:11px;color:#555'>(안정=SPY상승장+VIX안정/경계 · 위험=SPY하락장 or VIX공포/패닉)</span></div></div>
        </div>
      </div>
    </div>"""


# ------------------------------------------------------------------
def checklist_html(cur, row):
    conditions = [
        ("RSI 구간", cur["rsi_zone"], row["RSI구간"]),
        ("추세", cur["trend"], row["추세"]),
        ("BB 위치", cur["bb_zone"], row["BB위치"]),
        ("낙폭 구간", cur["dd_zone"], row["낙폭구간"]),
        ("MACD", cur["macd_dir"], row["MACD방향"]),
        ("ATR 변동성", cur["atr_zone"], row["ATR변동성"]),
        ("ADX 추세강도", cur["adx_zone"], row["ADX추세강도"]),
        ("장기추세", cur["long_trend"], row["장기추세"]),
        ("시장레짐", cur["market_regime"], row["시장레짐"]),
    ]
    n_cond = len(conditions)
    matched = sum(1 for _, c, t in conditions if c == t)
    bar_color = "#00897b" if matched == n_cond else "#f57c00" if matched >= 5 else "#c62828"

    rows_html = ""
    for label, c, t in conditions:
        ok = c == t
        bg, color = ("#1b5e20", "#e8f5e9") if ok else ("#3e2723", "#efebe9")
        tgt = f"<b style='color:#a5d6a7'>{t}</b>" if ok else f"<span style='color:#bcaaa4'>{t}</span>"
        rows_html += (f"<tr style='background:{bg};color:{color}'>"
                      f"<td style='padding:6px 12px;font-weight:bold;white-space:nowrap'>{'✅' if ok else '❌'} {label}</td>"
                      f"<td style='padding:6px 12px'>현재: <b>{c}</b></td>"
                      f"<td style='padding:6px 12px'>조합: {tgt}</td></tr>")

    grade = row["🏆등급"] if row["🏆등급"] else "(등급 없음)"
    gc = {"🏆 황금": "#ffca28", "★★ 신뢰": "#64b5f6", "💰 수익": "#81c784"}.get(row["🏆등급"], "#aaa")
    return f"""
    <div style='font-family:monospace;background:#212121;border-radius:10px;padding:16px;margin:10px 0;color:#eee'>
      <div style='font-size:15px;font-weight:bold;margin-bottom:10px'>
        조건 일치: {matched} / {n_cond} &nbsp;
        <span style='color:{bar_color};letter-spacing:3px'>{"█"*matched}{"░"*(n_cond-matched)}</span>
        &nbsp;&nbsp;<span style='color:{gc}'>{grade}</span>
      </div>
      {_scroll(f"<table style='border-collapse:collapse;width:100%;border-radius:6px;overflow:hidden'>{rows_html}</table>")}
      <div style='margin-top:12px;font-size:13px;color:#ccc;border-top:1px solid #444;padding-top:10px'>
        최적보유일 <b style='color:#fff'>{row["최적보유일"]}일</b> | 승률 <b style='color:#fff'>{row["승률%"]}%</b> |
        기대값 <b style='color:#fff'>{row["기대값점수"]}</b> | 유의성 <b style='color:#fff'>{row["유의성"]}</b> |
        샘플 <b style='color:#fff'>{row["횟수"]}회</b>
      </div>
    </div>"""


# ------------------------------------------------------------------
def gold_distribution_html(df_gold):
    if len(df_gold) == 0:
        return ""

    def dist(col, title):
        vc = df_gold[col].value_counts().sort_values(ascending=False)
        total = len(df_gold)
        rows = ""
        for val, cnt in vc.items():
            pct = cnt / total * 100
            rows += (f"<tr><td style='padding:4px 10px;color:#ccc;white-space:nowrap'>{val}</td>"
                     f"<td style='padding:4px 10px'><div style='background:#1565c0;height:14px;width:{int(pct*1.2)}px;border-radius:3px;display:inline-block'></div></td>"
                     f"<td style='padding:4px 10px;color:#90caf9;white-space:nowrap'>{cnt}개 ({pct:.0f}%)</td></tr>")
        return (f"<div style='margin-bottom:12px'><div style='color:#90caf9;font-size:12px;font-weight:bold;margin-bottom:4px'>{title}</div>"
                f"<table style='border-collapse:collapse;font-size:12px'>{rows}</table></div>")

    cols = [("RSI구간", "RSI구간"), ("추세", "추세"), ("BB위치", "BB위치"), ("낙폭구간", "낙폭구간"),
            ("MACD방향", "MACD"), ("ATR변동성", "ATR변동성"), ("ADX추세강도", "ADX추세강도"),
            ("장기추세", "장기추세"), ("시장레짐", "시장레짐")]
    half = (len(cols) + 1) // 2
    left = "".join(dist(c, t) for c, t in cols[:half])
    right = "".join(dist(c, t) for c, t in cols[half:])
    return f"""
    <div style='{WRAP}'>
      <div style='font-size:15px;font-weight:bold;color:#ffca28;margin-bottom:14px;border-bottom:1px solid #333;padding-bottom:8px'>
        🏆 황금 조합 분포 — 총 {len(df_gold)}개
        <span style='font-size:12px;font-weight:normal;color:#aaa'>(평균승률 {round(df_gold["승률%"].mean(),1)}% /
          평균기대값 {round(df_gold["기대값점수"].mean(),3)} / 최빈보유일 {df_gold["최적보유일"].value_counts().idxmax()}일)</span>
      </div>
      <div style='display:flex;flex-wrap:wrap;gap:0 24px'>
        <div style='flex:1 1 260px;min-width:0'>{left}</div>
        <div style='flex:1 1 260px;min-width:0'>{right}</div>
      </div>
    </div>"""


# ------------------------------------------------------------------
def prediction_card_html(n_hist, best_h, best_n, best_wr, best_ev, sig):
    hcol = "#00c853" if best_wr >= 0.7 else "#ffab00" if best_wr >= 0.5 else "#ef5350"
    return f"""
    <div style='font-family:monospace;background:#132a1c;border:1px solid {hcol};border-radius:10px;padding:20px;color:#eee;margin-top:8px'>
      <div style='font-size:22px;font-weight:bold;color:{hcol}'>{best_h}일 보유 시 승률 {round(best_wr*100,1)}% · 기대값 {round(best_ev,2)}</div>
      <div style='font-size:12px;color:#888;margin-top:8px'>과거 현재-조건 완전일치 사례 총 {n_hist}회 중, {best_h}일 보유 시점까지 데이터가 있는 {best_n}개 표본 기준 · 유의성 {sig}</div>
      <div style='font-size:11px;color:#666;margin-top:6px'>※ 위 "황금/신뢰" 등급표는 목표승률·최소샘플 조건을 만족한 조합만 모은 것이라, 지금 이 순간의 정확한 조건은 그 표에 없을 수도 있습니다. 이 섹션은 등급 문턱과 무관하게 현재 상태 그대로 계산한 실측치입니다.</div>
    </div>"""
