import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 한글 폰트 CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 데이터 읽기 (UTF-8-SIG → CP949 폴백)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# 열 이름 표준화
# ─────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ─────────────────────────────────────────────
# 학교명 추출 (파일명에서)
# ─────────────────────────────────────────────
def extract_school(filename):
    if not filename:
        return "○○고"
    base = os.path.splitext(os.path.basename(filename))[0]
    for sep in ("_", "-", " "):
        if sep in base:
            return base.split(sep)[0]
    return base[:12]

# ─────────────────────────────────────────────
# 숫자 열 stats 요약
# ─────────────────────────────────────────────
def num_stats(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        rows.append({
            "항목": c,
            "평균": round(s.mean(), 2),
            "최소": round(s.min(), 2),
            "최대": round(s.max(), 2),
            "표준편차": round(s.std(), 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 추세선 추가 헬퍼
# ─────────────────────────────────────────────
def add_trendline(fig, x, y, color="red", dash="dash", name="추세선"):
    mask = x.notna() & y.notna()
    x2 = x[mask]
    y2 = y[mask]
    if len(x2) < 2:
        return
    m, b = np.polyfit(x2, y2, 1)
    xline = np.array([x2.min(), x2.max()])
    fig.add_trace(go.Scatter(
        x=xline,
        y=m * xline + b,
        mode="lines",
        line=dict(color=color, dash=dash),
        name=name,
        showlegend=True,
    ))


# ─────────────────────────────────────────────
# 공통 Plotly 한글 폰트 적용
# ─────────────────────────────────────────────
def kfont(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        hoverlabel=dict(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")),
    )
    return fig


# ═════════════════════════════════════════════
# 페이지 설정
# ═════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ─────────────────────────────────────────────
# 사이드바: CSV 업로드
# ─────────────────────────────────────────────
st.sidebar.header("📁 CSV 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")
growth_item = st.sidebar.selectbox(
    "생육 측정 항목",
    ["생중량(g)", "잎 수", "지상부 길이(mm)", "지하부 길이(mm)"],
    index=0,
)

# 데이터는 맨 위에서 한 번만 읽음
env_df = pd.DataFrame()
growth_df = pd.DataFrame()
if env_file is not None:
    env_df = std_cols(read_csv(env_file))
if growth_file is not None:
    growth_df = std_cols(read_csv(growth_file))

sch_name = extract_school(
    env_file.name if env_file else (growth_file.name if growth_file else None)
)
st.sidebar.markdown(f"🏫 업로드된 학교: **{sch_name}**")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ═════════════════════════════════════════════
# Tab 1 — 실험 개요
# ═════════════════════════════════════════════
with tab1:
    try:
        st.markdown(f"### 🏫 {sch_name} 극지식물 실험 개요")

        st.markdown(
            """
**연구 배경 및 목적**

극지식물은 덥고 건조한 환경에서도 살아남는 강한 식물입니다.
이 실험에서는 우리 학교 실험실에서 키운 극지식물 개체들을 관찰하여
**개체마다 생육(잎 수, 줄기 길이, 무게 등)에 어떤 차이가 있는지**를 분석했습니다.

- 환경 데이터: 시간별 온도·습도·pH·EC 등을 기록
- 생육 결과 데이터: 각 개체의 잎 수, 지상부/지하부 길이, 생중량(무게)을 측정

위 두 데이터를 함께 보면 **어떤 환경에서 자란 개체가 잘 자랐는지**도 추론할 수 있습니다.
"""
        )

        if growth_df.empty and env_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 업로드해 주세요.")
        else:
            period_txt = "—"
            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    period_txt = f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}"

            col1, col2, col3, col4 = st.columns(4)
            n_individuals = len(growth_df) if not growth_df.empty else 0
            col1.metric("총 개체 수", f"{n_individuals}개체")

            avg_temp = "—"
            if not env_df.empty and "temperature" in env_df.columns:
                s = pd.to_numeric(env_df["temperature"], errors="coerce").dropna()
                if len(s) >= 2:
                    avg_temp = f"{s.mean():.1f}℃"
            col2.metric("평균 온도", avg_temp)

            avg_hum = "—"
            if not env_df.empty and "humidity" in env_df.columns:
                s = pd.to_numeric(env_df["humidity"], errors="coerce").dropna()
                if len(s) >= 2:
                    avg_hum = f"{s.mean():.1f}%"
            col3.metric("평균 습도", avg_hum)

            avg_bw = "—"
            if not growth_df.empty and "생중량" in growth_df.columns:
                s = pd.to_numeric(growth_df["생중량"], errors="coerce").dropna()
                if len(s) >= 2:
                    avg_bw = f"{s.mean():.2f}g"
            col4.metric("평균 생중량", avg_bw)

            st.markdown("---")
            st.subheader("📋 데이터 요약")

            summary_rows = []
            if not growth_df.empty:
                items = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                "지상부 질량", "지하부 질량"]
                    if c in growth_df.columns
                ]
                summary_rows.append({
                    "데이터": "생육 결과",
                    "개체 수": len(growth_df),
                    "측정 기간": period_txt,
                    "측정 항목": ", ".join(items),
                })
            if not env_df.empty:
                eitems = [
                    c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                    if c in env_df.columns
                ]
                summary_rows.append({
                    "데이터": "환경 데이터",
                    "개체 수": "—",
                    "측정 기간": period_txt,
                    "측정 항목": ", ".join(eitems),
                })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")
            else:
                st.info("표시할 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ═════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("사이드바에서 환경 데이터 CSV를 업로드해 주세요.")
        else:
            st.subheader("🌡️ 환경 데이터 요약")

            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                if c in env_df.columns
            ]
            if env_num_cols:
                st.dataframe(num_stats(env_df, env_num_cols), width="stretch")
            else:
                st.info("표시할 숫자 열이 없습니다.")

            st.markdown("---")
            st.subheader("📈 날짜별 환경 변화 (일별 평균)")

            daily = env_df[["time"]].copy() if "time" in env_df.columns else pd.DataFrame()
            if "time" in daily.columns and not daily.empty:
                daily["day"] = daily["time"].dt.floor("D")
                for c in ["temperature", "humidity", "ph", "ec"]:
                    if c in env_df.columns:
                        daily[c] = pd.to_numeric(env_df[c], errors="coerce")
                daily_g = daily.dropna(subset=["day"]).groupby("day").mean()
                daily_g = daily_g.sort_index()

                avail = [c for c in ["temperature", "humidity", "ph", "ec"]
                         if c in daily_g.columns]
                if avail:
                    n = len(avail)
                    fig = make_subplots(rows=n, cols=1, subplot_titles=avail)
                    colors = {
                        "temperature": "#e74c3c",
                        "humidity": "#3498db",
                        "ph": "#2ecc71",
                        "ec": "#9b59b6",
                    }
                    for i, c in enumerate(avail, 1):
                        fig.add_trace(
                            go.Scatter(
                                x=daily_g.index,
                                y=daily_g[c],
                                mode="lines",
                                name=c,
                                line=dict(color=colors.get(c, "#333"), width=2),
                                hovertemplate=f"{c}<br>%{{x|%Y-%m-%d}}<br>%{{y:.1f}}<extra></extra>",
                            ),
                            row=i, col=1,
                        )
                        fig.update_yaxes(title_text=c, row=i, col=1)
                    fig.update_xaxes(title_text="날짜", row=n, col=1)
                    fig.update_layout(
                        height=180 * n,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    kfont(fig)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("온도·습도·ph·ec 열이 모두 없습니다.")
            else:
                st.info("time 열이 없거나 날짜 형식이 아닙니다.")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{sch_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═════════════════════════════════════════════
# Tab 3 — 생육 결과
# ═════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            bw_col = "생중량"
            leaf_col = "잎 수"
            shoot_len_col = "지상부 길이"
            root_len_col = "지하부 길이"

            has_bw = bw_col in growth_df.columns
            has_leaf = leaf_col in growth_df.columns
            has_shoot = shoot_len_col in growth_df.columns
            has_root = root_len_col in growth_df.columns

            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")
            c1, c2, c3 = st.columns(3)

            if has_bw:
                bw = pd.to_numeric(growth_df[bw_col], errors="coerce").dropna()
                if len(bw) >= 2:
                    avg_bw_val = bw.mean()
                    max_idx = bw.idxmax()
                    min_idx = bw.idxmin()
                    c1.metric("평균 생중량", f"{avg_bw_val:.2f}g")
                    c2.metric("최대 생중량",
                              f"{bw.max():.2f}g",
                              delta=f"개체번호 {growth_df.loc[max_idx, '개체번호']}")
                    c3.metric("최소 생중량",
                              f"{bw.min():.2f}g",
                              delta=f"개체번호 {growth_df.loc[min_idx, '개체번호']}")
                else:
                    c1.metric("평균 생중량", "—")
                    c2.metric("최대 생중량", "—")
                    c3.metric("최소 생중량", "—")
            else:
                c1.metric("평균 생중량", "—")
                c2.metric("최대 생중량", "—")
                c3.metric("최소 생중량", "—")

            st.markdown("---")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")

            bar_targets = [
                ("평균 생중량", bw_col, "#27ae60"),
                ("평균 잎 수", leaf_col, "#2980b9"),
                ("평균 지상부 길이(mm)", shoot_len_col, "#e67e22"),
                ("평균 지하부 길이(mm)", root_len_col, "#8e44ad"),
            ]
            bar_avail = [(lbl, col, clr) for lbl, col, clr in bar_targets
                         if col in growth_df.columns]

            if bar_avail:
                rows_n = (len(bar_avail) + 1) // 2
                fig_bar = make_subplots(rows=rows_n, cols=2,
                                        subplot_titles=[b[0] for b in bar_avail])
                for i, (lbl, col, clr) in enumerate(bar_avail):
                    s = pd.to_numeric(growth_df[col], errors="coerce")
                    row_i = (i // 2) + 1
                    col_i = (i % 2) + 1
                    fig_bar.add_trace(
                        go.Bar(
                            x=growth_df.loc[s.dropna().index, "개체번호"].astype(str).tolist(),
                            y=s.dropna().values,
                            name=lbl,
                            marker_color=clr,
                            hovertemplate=f"{lbl}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>",
                        ),
                        row=row_i, col=col_i,
                    )
                    fig_bar.update_yaxes(title_text=lbl, row=row_i, col=col_i)
                fig_bar.update_xaxes(title_text="개체번호", row=rows_n, col=2)
                fig_bar.update_layout(
                    height=280 * rows_n,
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                kfont(fig_bar)
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("표시할 생육 항목이 없습니다.")

            st.markdown("---")

            # ── 생중량 분포 (박스플롯) ──
            if has_bw:
                bw = pd.to_numeric(growth_df[bw_col], errors="coerce").dropna()
                if len(bw) >= 2:
                    q1v = bw.quantile(0.25)
                    q3v = bw.quantile(0.75)
                    iqr = q3v - q1v
                    upper_fence = q3v + 1.5 * iqr
                    lower_fence = q1v - 1.5 * iqr

                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=bw.values,
                        name="생중량 분포",
                        fillcolor="rgba(39, 174, 96, 0.35)",
                        line=dict(color="#27ae60", width=2),
                        whiskerwidth=0.8,
                        boxpoints="outliers",
                        hovertemplate="생중량<br>%{y:.2f}g<extra></extra>",
                    ))
                    fig_box.add_hline(y=upper_fence, line_dash="dot",
                                      annotation_text=f"상위 경계 (Q3+1.5×IQR = {upper_fence:.2f}g)",
                                      annotation_position="top right")
                    fig_box.add_hline(y=lower_fence, line_dash="dot",
                                      annotation_text=f"하위 경계 (Q1-1.5×IQR = {lower_fence:.2f}g)",
                                      annotation_position="bottom right")
                    fig_box.update_layout(
                        title="📦 생중량 분포 (박스플롯) — 상자: Q1~Q3, 굵은 선: 중앙값, 점: 이상치",
                        yaxis_title="생중량 (g)",
                        margin=dict(l=40, r=20, t=60, b=40),
                        height=360,
                    )
                    kfont(fig_box)
                    st.plotly_chart(fig_box, width="stretch")

                    cv = (bw.std() / bw.mean()) * 100 if bw.mean() > 0 else 0
                    st.markdown(
                        f"""
**📐 분포 지표 (생중량 기준, n={len(bw)}개체)**
- **평균**: {bw.mean():.2f}g
- **표준편차**: {bw.std():.2f}g  
  *(표준편차: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 지표)*
- **CV(변동계수)**: {cv:.1f}%  
  *(CV: 표준편차를 평균으로 나눈 값, 단위가 다른 집단 간 비교에 유리)*
- **사분위 범위(IQR)**: Q1={q1v:.2f}g ~ Q3={q3v:.2f}g
"""
                    )
                else:
                    st.info("생중량 값이 충분하지 않아 분포를 그리지 못했어요.")
            else:
                st.info("생중량 열이 없습니다.")

            st.markdown("---")

            # ── 상관관계 분석 ──
            st.subheader("🔗 상관관계 분석")

            corr_pairs = [
                ("잎 수 vs 생중량", leaf_col, bw_col, "#2980b9"),
                ("지상부 길이 vs 생중량", shoot_len_col, bw_col, "#e67e22"),
            ]
            corr_avail = [(lbl, xcol, ycol, clr) for lbl, xcol, ycol, clr in corr_pairs
                          if xcol in growth_df.columns and ycol in growth_df.columns]

            if corr_avail:
                rows_c = len(corr_avail)
                fig_corr = make_subplots(rows=rows_c, cols=1,
                                         subplot_titles=[c[0] for c in corr_avail])
                for i, (lbl, xcol, ycol, clr) in enumerate(corr_avail):
                    x = pd.to_numeric(growth_df[xcol], errors="coerce")
                    y = pd.to_numeric(growth_df[ycol], errors="coerce")
                    mask = x.notna() & y.notna()
                    x2 = x[mask]
                    y2 = y[mask]
                    if len(x2) < 2:
                        continue
                    r = np.corrcoef(x2, y2)[0, 1]
                    fig_corr.add_trace(
                        go.Scatter(
                            x=x2,
                            y=y2,
                            mode="markers",
                            marker=dict(color=clr, size=8, opacity=0.7,
                                       line=dict(width=1, color="white")),
                            name=f"r = {r:.3f}",
                            hovertemplate=f"{lbl}<br>%{{x}} | %{{y:.2f}}<extra></extra>",
                        ),
                        row=i + 1, col=1,
                    )
                    add_trendline(fig_corr, x2, y2, color="red", dash="dash", name="추세선")
                    fig_corr.update_xaxes(title_text=xcol, row=i + 1, col=1)
                    fig_corr.update_yaxes(title_text=ycol, row=i + 1, col=1)
                fig_corr.update_layout(
                    height=300 * rows_c,
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=True,
                )
                kfont(fig_corr)
                st.plotly_chart(fig_corr, width="stretch")

                for lbl, xcol, ycol, clr in corr_avail:
                    x = pd.to_numeric(growth_df[xcol], errors="coerce")
                    y = pd.to_numeric(growth_df[ycol], errors="coerce")
                    mask = x.notna() & y.notna()
                    if mask.sum() < 2:
                        continue
                    r = np.corrcoef(x[mask], y[mask])[0, 1]
                    strength = (
                        "강한 양의 상관" if r >= 0.7
                        else "중간 양의 상관" if r >= 0.4
                        else "약한 상관" if r >= -0.4
                        else "중간 음의 상관" if r >= -0.7
                        else "강한 음의 상관"
                    )
                    st.markdown(
                        f"""
**{lbl}**
- **상관계수(r)**: {r:.4f} → {strength}
- *(상관계수 r: -1 ~ +1 사이 값, +1에 가까울수록 함께 증가, -1에 가까울수록 한쪽 증가 시 다른 쪽 감소)*
"""
                    )
            else:
                st.info("산점도를 그릴 수 있는 열이 충분하지 않아요.")

            st.markdown("---")

            # ── 생중량 상위 25% vs 하위 25% 비교 ──
            if has_bw:
                bw = pd.to_numeric(growth_df[bw_col], errors="coerce").dropna()
                if len(bw) >= 4:
                    q1v = bw.quantile(0.25)
                    q3v = bw.quantile(0.75)
                    top_idx = bw[bw >= q3v]
                    bot_idx = bw[bw <= q1v]

                    st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")

                    comp_rows = []
                    if has_leaf:
                        top_leaf = pd.to_numeric(
                            growth_df.loc[top_idx.index, "잎 수"], errors="coerce"
                        ).mean()
                        bot_leaf = pd.to_numeric(
                            growth_df.loc[bot_idx.index, "잎 수"], errors="coerce"
                        ).mean()
                        comp_rows.append({"그룹": "상위 25%",
                                          "평균 생중량(g)": round(top_idx.mean(), 2),
                                          "평균 잎 수": round(top_leaf, 1)})
                        comp_rows.append({"그룹": "하위 25%",
                                          "평균 생중량(g)": round(bot_idx.mean(), 2),
                                          "평균 잎 수": round(bot_leaf, 1)})
                    if has_shoot:
                        top_sl = pd.to_numeric(
                            growth_df.loc[top_idx.index, "지상부 길이"], errors="coerce"
                        ).mean()
                        bot_sl = pd.to_numeric(
                            growth_df.loc[bot_idx.index, "지상부 길이"], errors="coerce"
                        ).mean()
                        comp_rows.append({"그룹": "상위 25%",
                                          "평균 지상부 길이(mm)": round(top_sl, 1)})
                        comp_rows.append({"그룹": "하위 25%",
                                          "평균 지상부 길이(mm)": round(bot_sl, 1)})
                    if has_root:
                        top_rl = pd.to_numeric(
                            growth_df.loc[top_idx.index, "지하부 길이"], errors="coerce"
                        ).mean()
                        bot_rl = pd.to_numeric(
                            growth_df.loc[bot_idx.index, "지하부 길이"], errors="coerce"
                        ).mean()
                        comp_rows.append({"그룹": "상위 25%",
                                          "평균 지하부 길이(mm)": round(top_rl, 1)})
                        comp_rows.append({"그룹": "하위 25%",
                                          "평균 지하부 길이(mm)": round(bot_rl, 1)})

                    if comp_rows:
                        st.dataframe(pd.DataFrame(comp_rows), width="stretch")

                    st.markdown("**상위 25% 개체 목록**")
                    top_list = growth_df.loc[top_idx.index].copy()
                    top_list["생중량"] = pd.to_numeric(top_list["생중량"], errors="coerce")
                    top_list = top_list.sort_values("생중량", ascending=False)
                    st.dataframe(
                        top_list[["개체번호", "생중량"]].reset_index(drop=True),
                        width="stretch",
                    )
                    st.markdown("**하위 25% 개체 목록**")
                    bot_list = growth_df.loc[bot_idx.index].copy()
                    bot_list["생중량"] = pd.to_numeric(bot_list["생중량"], errors="coerce")
                    bot_list = bot_list.sort_values("생중량", ascending=True)
                    st.dataframe(
                        bot_list[["개체번호", "생중량"]].reset_index(drop=True),
                        width="stretch",
                    )
                else:
                    st.info("개체 수가 충분하지 않아 상위/하위 25% 비교를 못해요.")
            else:
                st.info("생중량 열이 없어 상위/하위 비교를 못해요.")

            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="생육 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{sch_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
