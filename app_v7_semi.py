# -*- coding: utf-8 -*-
"""SOXL 승률 조건 역탐색 v7 (반도체판) — Streamlit 앱

원본 app.py(v6)는 그대로 두고 새로 만든 파일입니다.
실행:  streamlit run app_v7_semi.py
변경: BB → 거래량 비율 / 시장레짐 = 반도체 상대강도 + VIX / 🔗 상관 진단 탭 추가
"""

import numpy as np
import pandas as pd
import streamlit as st

import engine_v7_semi as engine
import views_v7_semi as views
from indicators import calc_ev, sig_label

st.set_page_config(
    page_title="SOXL 역탐색 v7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 다크 톤 + 모바일 여백 최적화
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem; max-width: 1200px;}
  @media (max-width: 640px) { h1 {font-size: 1.4rem !important;} }
</style>
""", unsafe_allow_html=True)


def html(s):
    if s:
        st.markdown(s, unsafe_allow_html=True)


# ------------------------------------------------------------------
# 캐시: 데이터 로드(1시간) / 조합 탐색(같은 설정이면 재사용)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load(years):
    return engine.load_and_build(years)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(years, threshold_pct, min_n, crit):
    data, close, ctx = cached_load(years)
    return engine.search_combinations(data, close, ctx, threshold_pct, min_n, crit)


# ------------------------------------------------------------------
# 사이드바 (설정)
# ------------------------------------------------------------------
st.title("📈 SOXL 역탐색 v7 · 반도체판")
st.caption("RSI · 추세 · 거래량 · 낙폭 · MACD · ATR · ADX · 장기추세 · 시장레짐(반도체상대강도·VIX) — 9개 조건 조합 백테스트")

with st.sidebar:
    st.header("⚙️ 설정")
    years = st.selectbox("데이터 기간(년)", [5, 10, 15], index=1)
    win_thresh = st.slider("목표승률 %", 60, 95, 75, 1)
    min_samples = st.slider("최소 샘플수", 5, 50, 15, 5)
    crit = st.selectbox("최적 기준", ["기대값점수", "승률%", "Sharpe", "평균수익%"], index=0)
    sort_col = st.selectbox("정렬 기준", ["🏆등급", "승률%", "기대값점수", "Sharpe", "평균수익%", "횟수"], index=0)
    top_n = st.selectbox("최대 표시 수", [10, 20, 30, 50, 999], index=2)
    run_btn = st.button("🔍 탐색 실행", type="primary", width='stretch')
    if st.button("🔄 데이터 새로고침", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("데이터: Yahoo Finance (SOXL / SOXX / QQQ / SPY / ^VIX)")

# 설정이 바뀌어도 자동으로 다시 돌리지 않고, 버튼을 눌렀을 때만 실행
sig = (years, win_thresh, min_samples, crit)
if run_btn:
    st.session_state["ran_sig"] = sig
ran_sig = st.session_state.get("ran_sig")

if ran_sig is None:
    st.info("👈 왼쪽(모바일은 좌상단 ☰) 설정에서 **탐색 실행**을 눌러주세요.")
    with st.expander("📖 지표 용어 상세 설명", expanded=False):
        html(views.glossary_html())
    st.stop()

years, win_thresh, min_samples, crit = ran_sig

# ------------------------------------------------------------------
# 데이터 + 현재 상태
# ------------------------------------------------------------------
try:
    with st.spinner("데이터 로딩 및 지표 계산 중..."):
        data, close, ctx = cached_load(years)
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

cur = engine.current_state(data, close, ctx)
st.success(f"총 {len(data):,}거래일 로드 · 최근 데이터 {data.index[-1].strftime('%Y-%m-%d')} · SOXL {cur['price']:.2f}")

# ------------------------------------------------------------------
# 조합 탐색
# ------------------------------------------------------------------
key = (years, win_thresh, min_samples, crit)
with st.spinner(f"조합 탐색 중 (목표승률 {win_thresh}% / 최소샘플 {min_samples} / 기준 {crit})..."):
    df = cached_search(*key)

if df is None or len(df) == 0:
    st.warning("조건을 만족하는 조합이 없습니다. 목표승률을 낮추거나 최소샘플수를 줄여보세요.")
    tab_state, tab_gloss = st.tabs(["📊 현재 상태", "📖 용어"])
    with tab_state:
        html(views.market_state_html(cur))
    with tab_gloss:
        html(views.glossary_html())
    st.stop()

df = df.copy()
df = engine.assign_grades(df, crit, top_n)
df["_match"] = engine.count_match(df, cur)

grade_order = {"🏆 황금": 0, "★★ 신뢰": 1, "💰 수익": 2, "": 3}
if sort_col == "🏆등급":
    df["_gr"] = df["🏆등급"].map(grade_order)
    df = df.sort_values(["_gr", "기대값점수"], ascending=[True, False]).drop(columns=["_gr"])
else:
    df = df.sort_values(sort_col, ascending=False)
df = df.reset_index(drop=True)

gold_n = int((df["🏆등급"] == "🏆 황금").sum())
sig_n = int((df["🏆등급"] == "★★ 신뢰").sum())
prof_n = int((df["🏆등급"] == "💰 수익").sum())

# ------------------------------------------------------------------
# 탭 구성
# ------------------------------------------------------------------
tab_now, tab_match, tab_table, tab_gold, tab_state, tab_corr, tab_gloss = st.tabs([
    "🎯 지금 진입", "📍 패턴 매칭", "📋 탐색 결과", "🏆 황금 분포", "📊 현재 상태", "🔗 상관 진단", "📖 용어",
])

# ---- 🎯 지금 진입 ----
with tab_now:
    st.subheader("지금 진입 시 예측")
    st.caption("현재 9개 조건과 정확히 같았던 과거 사례만으로 계산한 실측치 (등급 문턱과 무관)")

    full_mask = engine.current_full_mask(data, cur)
    n_hist = int(full_mask.sum())

    cands = []
    for h in engine.SHOW_HOLDS:
        r = (close.shift(-h) / close - 1).reindex(data.index)[full_mask].dropna()
        if len(r) < 5:
            continue
        cands.append((h, len(r), (r > 0).mean(), calc_ev(r), r))

    if n_hist < 5 or not cands:
        st.warning(f"현재 9개 조건과 정확히 같았던 과거 사례가 {n_hist}회뿐입니다 "
                   f"(보유일별 최소 5회 필요). 표본이 부족해 신뢰할 만한 예측을 만들기 어렵습니다 — "
                   f"'📍 패턴 매칭' 탭의 근접 조합을 참고하세요.")
    else:
        valid = [c for c in cands if c[3] is not None and not pd.isna(c[3])]
        pool = valid if valid else cands
        best_h, best_n, best_wr, best_ev, best_r = max(
            pool, key=lambda c: c[3] if c[3] is not None and not pd.isna(c[3]) else -999)
        s = sig_label(int((best_r > 0).sum()), len(best_r))
        html(views.prediction_card_html(n_hist, best_h, best_n, best_wr, best_ev, s))
        html(views.holding_analysis_html(data, close, full_mask,
             "지금 진입 시 보유일별 승률·기대값 예측 (현재 조건 완전일치 기준)"))
        html(views.sample_details_html(data, close, full_mask, hold_days=best_h,
             title="현재 조건과 정확히 같았던 과거 사례 전체"))

# ---- 📍 패턴 매칭 ----
with tab_match:
    st.subheader("현재 상태 패턴 매칭")
    st.caption(f"{cur['rsi_zone']} / {cur['trend']} / {cur['vol_zone']} / {cur['dd_zone']} / "
               f"{cur['macd_dir']} / {cur['atr_zone']} / {cur['adx_zone']} / "
               f"{cur['long_trend']} / {cur['market_regime']}")

    def render_combo(row, label):
        html(views.checklist_html(cur, row))
        mask = engine.build_cond_mask(data, row["_cond_key"])
        bh = int(row["최적보유일"])
        html(views.holding_analysis_html(data, close, mask, f"{label} 보유일별 예상 성과"))
        html(views.yearly_stats_html(data, close, mask, bh, f"{label} 연도별 승률 (보유일 {bh}일)"))
        html(views.period_comparison_html(data, close, mask, bh))
        html(views.sample_details_html(data, close, mask, hold_days=bh,
             title=f"{label} — 개별 샘플 상세"))

    full = df[df["_match"] == 9]
    if len(full) > 0:
        m = full.iloc[0]
        msg = {
            "🏆 황금": ("success", "🟢 지금이 기회입니다! — 황금 패턴 완전 일치"),
            "★★ 신뢰": ("warning", "🟡 주목할 구간 — 신뢰 패턴 완전 일치"),
            "💰 수익": ("info", "🟡 수익성 패턴 완전 일치 (통계 검증 약함)"),
        }.get(m["🏆등급"], ("info", "⚪ 완전 일치하나 황금/신뢰 등급 아님"))
        getattr(st, msg[0])(msg[1])
        render_combo(m, "완전 일치 조합")
    else:
        st.warning("⚠️ 완전 일치 패턴 없음 — 현재 상태와 가장 근접한 조합 Top 3")
        tmp = df.copy()
        tmp["_gr"] = tmp["🏆등급"].map(grade_order).fillna(3)
        top3 = tmp.sort_values(["_match", "_gr", "기대값점수"], ascending=[False, True, False]).head(3)
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            with st.expander(f"#{rank} — {int(row['_match'])}/9 조건 일치", expanded=(rank == 1)):
                render_combo(row, f"#{rank} 조합")

# ---- 📋 탐색 결과 ----
with tab_table:
    st.subheader(f"탐색 결과 — 총 {len(df):,}개 유효 조합")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 황금", gold_n)
    c2.metric("★★ 신뢰", sig_n)
    c3.metric("💰 수익", prof_n)

    show_cols = [
        "🏆등급", "RSI구간", "추세", "거래량", "낙폭구간", "MACD방향", "ATR변동성",
        "ADX추세강도", "장기추세", "시장레짐", "최적보유일", "횟수", "승률%",
        "평균수익%", "중앙값수익%", "손실평균%", "Sharpe", "MDD%", "기대값점수", "유의성",
        "평균RSI", "평균MA이격%", "평균거래량비", "평균낙폭%", "평균ATR%", "평균ADX", "유효보유일범위",
    ]
    n_show = len(df) if top_n == 999 else top_n
    st.dataframe(df[show_cols].head(n_show), width='stretch', hide_index=True, height=520)

    csv = df[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", csv, file_name="soxl_reverse_search.csv", mime="text/csv")

# ---- 🏆 황금 분포 ----
with tab_gold:
    gold_df = df[df["🏆등급"] == "🏆 황금"]
    if len(gold_df) == 0:
        st.info("황금 등급 조합이 없습니다.")
    else:
        html(views.gold_distribution_html(gold_df))

# ---- 📊 현재 상태 ----
with tab_state:
    html(views.market_state_html(cur))

# ---- 🔗 상관 진단 ----
with tab_corr:
    st.subheader("🔗 반도체는 시장과 얼마나 같이 움직이나")
    st.caption(f"SOXX(반도체 ETF)와 시장 ETF의 일간 수익률 상관, {engine.CORR_WINDOW}거래일 롤링. "
               "1에 가까울수록 같이 움직이고, 낮을수록 따로 논다는 뜻입니다.")
    cdf = ctx.get("corr_df")
    if cdf is None or len(cdf) == 0:
        st.info("상관을 계산할 데이터가 부족합니다.")
    else:
        summ = engine.corr_summary(cdf)
        cols = st.columns(len(summ))
        for col, (name, v) in zip(cols, summ.items()):
            col.metric(name, f"{v['current']:.2f}",
                       f"전체평균 {v['mean_all']:.2f} · 최근1년 {v['mean_1y']:.2f}", delta_color="off")
            col.caption(f"과거 분포의 하위 {v['pctile']:.0f}% 수준")
        st.line_chart(cdf, height=260)

        st.markdown("**반도체 상대강도** (SOXX÷QQQ 비율의 20일 변화, %) — 시장레짐 조건에 쓰이는 값")
        rs_hist = data["semi_rs"].iloc[-500:].rename("상대강도%")
        st.line_chart(rs_hist, height=200)
        st.caption(f"현재 {cur['semi_rs']:+.2f}% · {engine.RS_WEAK:+.0f}% 미만이면 '반도체약세'로 분류됩니다.")

# ---- 📖 용어 ----
with tab_gloss:
    html(views.glossary_html())

st.caption("※ 장중에 실행하면 마지막 봉(가격·거래량)이 아직 진행 중이라 값이 달라질 수 있습니다.")
st.caption("※ 과거 통계 기반 참고 자료이며 미래 수익을 보장하지 않습니다. 레버리지 ETF(SOXL)는 변동성이 매우 큽니다.")
