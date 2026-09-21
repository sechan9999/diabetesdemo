"""교란 유무에 따라 분석 방법의 성능이 어떻게 갈리는지 보여주는 대화형 데모."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="Diabetes Drug Trial: Methods", layout="wide")

# ── 팔레트 ────────────────────────────────────────────
# dataviz 기준 검증 통과(light/dark 전 쌍). 슬롯 1·2·3만 사용한다.
LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
         "surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f"}

try:
    _is_dark = st.context.theme.type == "dark"
except Exception:
    _is_dark = False
PAL = DARK if _is_dark else LIGHT


def style(fig, height=340):
    """모든 그림에 공통으로 적용하는 축·배경·여백 설정."""
    fig.update_layout(
        height=height, paper_bgcolor=PAL["surface"], plot_bgcolor=PAL["surface"],
        font=dict(color=PAL["ink"], size=13), margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=PAL["grid"], zeroline=False, linecolor=PAL["grid"],
                     title_font=dict(color=PAL["muted"]), tickfont=dict(color=PAL["muted"]))
    fig.update_yaxes(gridcolor=PAL["grid"], zeroline=False, linecolor=PAL["grid"],
                     title_font=dict(color=PAL["muted"]), tickfont=dict(color=PAL["muted"]))
    return fig


def combo_ci(res, weights, alpha=0.05):
    """선형결합의 추정치와 신뢰구간. treated + treated:week*26 같은 합을 다룬다."""
    # MixedLM의 params에는 Group Var가 섞여 있으므로 고정효과 이름만 쓴다.
    # t_test는 2차원 대비행렬을 요구한다.
    names = list(res.model.exog_names)
    c = np.zeros((1, len(names)))
    for k, w in weights.items():
        c[0, names.index(k)] = w
    tt = res.t_test(c)
    ci = tt.conf_int(alpha=alpha)
    return float(np.ravel(tt.effect)[0]), float(ci[0, 0]), float(ci[0, 1])


# ── 시나리오 A: 교란 없음 ─────────────────────────────
@st.cache_data(show_spinner=False)
def run_clean(n, true_effect, cutoff, bandwidth, seed):
    rng = np.random.default_rng(seed)
    baseline = rng.normal(7.5, 1.0, n)
    treated = (baseline >= cutoff).astype(int)
    pid = np.arange(n)
    # analysis.py와 난수 스트림을 맞추기 위해 age·bmi도 같은 순서로 뽑는다.
    # 기본 설정에서 README의 숫자가 그대로 재현된다.
    age = rng.normal(55, 10, n)
    bmi = rng.normal(31, 4, n)
    re_i = rng.normal(0, 0.5, n)

    def visit(week, decay):
        return pd.DataFrame({
            "patient_id": pid, "week": week,
            "hba1c": baseline + decay * (true_effect * treated) + re_i + rng.normal(0, 0.4, n),
            "baseline_hba1c": baseline, "treated": treated, "age": age, "bmi": bmi,
        })

    long = pd.concat([visit(0, 0.0), visit(12, 0.6), visit(26, 1.0)], ignore_index=True)
    w26 = long[long.week == 26].copy()

    m_ols = smf.ols("hba1c ~ treated + baseline_hba1c", w26).fit()
    ols = (m_ols.params["treated"], *m_ols.conf_int().loc["treated"].tolist())

    w26["centered"] = w26["baseline_hba1c"] - cutoff
    win = w26[w26["centered"].abs() <= bandwidth].copy()
    m_rdd = smf.ols("hba1c ~ treated * centered", win).fit()
    rdd = (m_rdd.params["treated"], *m_rdd.conf_int().loc["treated"].tolist())

    m_lin = smf.mixedlm("hba1c ~ treated * week + baseline_hba1c", long,
                        groups=long["patient_id"]).fit()
    lin = combo_ci(m_lin, {"treated": 1.0, "treated:week": 26.0})

    post = long[long.week > 0].copy()
    m_mmrm = smf.mixedlm("hba1c ~ treated * C(week) + baseline_hba1c", post,
                         groups=post["patient_id"]).fit()
    mmrm = combo_ci(m_mmrm, {"treated": 1.0, "treated:C(week)[T.26]": 1.0})

    traj = long.groupby(["week", "treated"])["hba1c"].mean().reset_index()
    return {
        "rows": [("단순 회귀", *ols, len(w26)), ("RDD", *rdd, len(win)),
                 ("혼합효과 (선형 시간)", *lin, len(w26)),
                 ("혼합효과 (MMRM)", *mmrm, len(w26))],
        "scatter": w26.sample(min(700, len(w26)), random_state=0),
        "traj": traj, "n_window": len(win),
        "summaries": {"단순 회귀": m_ols.summary().tables[1].as_text(),
                      "RDD": m_rdd.summary().tables[1].as_text(),
                      "혼합효과 (선형 시간)": str(m_lin.summary().tables[1]),
                      "혼합효과 (MMRM)": str(m_mmrm.summary().tables[1])},
    }


# ── 시나리오 B: 미관측 교란 있음 ──────────────────────
@st.cache_data(show_spinner=False)
def run_confounded(n, true_effect, cutoff, bandwidth, gamma, delta, n_boot, seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, n)
    baseline = 7.5 + 0.5 * U + rng.normal(0, 0.8, n)
    above = (baseline >= cutoff).astype(int)
    treated = rng.binomial(1, np.clip(0.15 + 0.70 * above + gamma * U, 0.01, 0.99))
    hba1c = baseline + true_effect * treated + delta * U + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"hba1c": hba1c, "treated": treated, "baseline_hba1c": baseline,
                       "above": above, "centered": baseline - cutoff, "U": U})

    m_naive = smf.ols("hba1c ~ treated + baseline_hba1c", df).fit()
    naive = (m_naive.params["treated"], *m_naive.conf_int().loc["treated"].tolist())
    m_orac = smf.ols("hba1c ~ treated + baseline_hba1c + U", df).fit()
    orac = (m_orac.params["treated"], *m_orac.conf_int().loc["treated"].tolist())

    win = df[df["centered"].abs() <= bandwidth].copy()
    fs = smf.ols("treated ~ above * centered", win).fit()
    rf = smf.ols("hba1c ~ above * centered", win).fit()
    late = rf.params["above"] / fs.params["above"]

    idx = np.arange(len(win))
    boot = []
    for _ in range(n_boot):
        s = win.iloc[rng.choice(idx, len(idx), replace=True)]
        b_fs = smf.ols("treated ~ above * centered", s).fit().params["above"]
        if abs(b_fs) < 1e-6:
            continue
        boot.append(smf.ols("hba1c ~ above * centered", s).fit().params["above"] / b_fs)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)

    return {
        "rows": [("단순 회귀 (U 미관측)", *naive, len(df)),
                 ("오라클 회귀 (U 관측)", *orac, len(df)),
                 ("퍼지 RDD", late, lo, hi, len(win))],
        "corr": float(np.corrcoef(treated, U)[0, 1]),
        "first_stage": float(fs.params["above"]), "reduced_form": float(rf.params["above"]),
        "scatter": df.sample(min(700, len(df)), random_state=0), "n_window": len(win),
        "summaries": {"단순 회귀 (U 미관측)": m_naive.summary().tables[1].as_text(),
                      "오라클 회귀 (U 관측)": m_orac.summary().tables[1].as_text(),
                      "퍼지 RDD 1단계": fs.summary().tables[1].as_text(),
                      "퍼지 RDD 축약형": rf.summary().tables[1].as_text()},
    }


def forest(rows, true_effect, title):
    """추정치와 신뢰구간. 값은 직접 라벨로 찍는다(대비 완화 규칙)."""
    labels = [r[0] for r in rows][::-1]
    est = [r[1] for r in rows][::-1]
    lo = [r[1] - r[2] for r in rows][::-1]
    hi = [r[3] - r[1] for r in rows][::-1]
    colors = [PAL["s1"], PAL["s2"], PAL["s3"], PAL["s1"]][:len(rows)][::-1]

    fig = go.Figure()
    fig.add_vline(x=true_effect, line=dict(color=PAL["muted"], width=2, dash="dash"),
                  annotation_text=f"참값 {true_effect:.2f}",
                  annotation_position="top", annotation_yshift=6,
                  annotation_font=dict(color=PAL["muted"], size=12))
    fig.add_trace(go.Scatter(
        x=est, y=labels, mode="markers", marker=dict(size=13, color=colors,
        line=dict(width=2, color=PAL["surface"])),
        error_x=dict(type="data", array=hi, arrayminus=lo, color=PAL["muted"], width=6, thickness=2),
        showlegend=False,
        hovertemplate="%{y}<br>추정치 %{x:.3f}<extra></extra>"))
    span_lo = min(min(r[2] for r in rows), true_effect)
    span_hi = max(max(r[3] for r in rows), true_effect)
    pad = max(0.12, (span_hi - span_lo) * 0.10)
    # 값은 오른쪽 끝에 한 열로 정렬한다. 대비 완화 규칙상 직접 라벨은 유지한다.
    fig.add_trace(go.Scatter(
        x=[span_hi + pad * 1.35] * len(est), y=labels, mode="text",
        text=[f"{v:.3f}" for v in est], textposition="middle right",
        textfont=dict(color=PAL["ink"], size=13), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title)
    # 오른쪽 여백을 더 크게 잡아 값 라벨이 들어갈 자리를 만든다.
    fig.update_xaxes(title="26주차 처치효과 (%p)",
                     range=[span_lo - pad, span_hi + pad * 3.2])
    fig.update_yaxes(title=None)
    return style(fig, height=300)


# ══ 화면 ══════════════════════════════════════════════
st.title("교란이 있을 때와 없을 때, 방법은 어떻게 갈리는가")
st.caption("같은 질문에 네 가지 방법으로 답합니다. 참값을 미리 고정했으므로 "
           "각 방법이 그 값을 얼마나 잘 회복하는지 직접 비교할 수 있습니다.")

with st.sidebar:
    st.header("시뮬레이션 설정")
    true_effect = st.slider("참 처치효과 (%p)", -2.5, 0.0, -1.2, 0.1)
    n = st.select_slider("표본 수 (교란 없음)", [500, 1000, 2000, 4000, 8000], 2000)
    cutoff = st.slider("처방 기준선 (HbA1c %)", 6.5, 8.0, 7.0, 0.1)
    bandwidth = st.slider("RDD 대역폭", 0.25, 1.5, 0.5, 0.05)
    st.divider()
    st.header("교란 설정")
    st.caption("U는 차트에 기록되지 않는 중증도입니다.")
    n_conf = st.select_slider("표본 수 (교란 있음)", [2000, 4000, 8000, 16000], 8000,
                              help="퍼지 RDD는 Wald 비율이라 대역폭 안 표본이 얇으면 "
                                   "추정이 불안정합니다. 8000 이상을 권합니다.")
    gamma = st.slider("U → 처방 (교란 강도)", 0.0, 0.40, 0.12, 0.01)
    delta = st.slider("U → 결과", 0.0, 1.5, 0.8, 0.1)
    n_boot = st.select_slider("부트스트랩 반복", [100, 200, 300, 500, 1000], 300)
    seed = st.number_input("난수 시드", 0, 9999, 42)

with st.spinner("모형 적합 중..."):
    A = run_clean(n, true_effect, cutoff, bandwidth, seed)
    B = run_confounded(n_conf, true_effect, cutoff, bandwidth, gamma, delta, n_boot, seed)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("교란 없음")
    st.caption("처방이 관측된 기저치만의 함수. 단순 회귀가 맞는 것은 설계상 당연합니다.")
    for name, e, _, _, k in A["rows"]:
        st.metric(name, f"{e:.3f}", f"{e - true_effect:+.3f} 편향",
                  delta_color="off", help=f"표본 {k:,}")
with right:
    st.subheader("미관측 교란 있음")
    st.caption(f"U가 처방과 결과에 동시에 작용합니다. 표본 {n_conf:,}. "
               f"corr(처방, U) = {B['corr']:.3f}")
    for name, e, _, _, k in B["rows"]:
        st.metric(name, f"{e:.3f}", f"{e - true_effect:+.3f} 편향",
                  delta_color="off", help=f"표본 {k:,}")

st.divider()
f1, f2 = st.columns(2, gap="large")
f1.plotly_chart(forest(A["rows"], true_effect, "교란 없음"), use_container_width=True)
f2.plotly_chart(forest(B["rows"], true_effect, "교란 있음"), use_container_width=True)

naive_bias = abs(B["rows"][0][1] - true_effect)
rdd_bias = abs(B["rows"][2][1] - true_effect)
if naive_bias > rdd_bias * 1.5:
    st.success(f"교란이 있으면 단순 회귀 편향({naive_bias:.3f})이 퍼지 RDD "
               f"편향({rdd_bias:.3f})보다 큽니다. 준실험 설계가 표본을 줄이는 비용을 정당화합니다.")
else:
    st.info(f"현재 설정에서는 두 방법의 편향 차이가 크지 않습니다 "
            f"(단순 회귀 {naive_bias:.3f}, 퍼지 RDD {rdd_bias:.3f}). "
            f"교란 강도 슬라이더를 올리면 차이가 드러납니다.")

st.divider()
st.subheader("기준선에서의 불연속")
s1, s2 = st.columns(2, gap="large")
for col, data, key, title in [(s1, A["scatter"], "treated", "교란 없음: 처방이 결정적"),
                              (s2, B["scatter"], "treated", "교란 있음: 처방이 확률적")]:
    fig = go.Figure()
    for val, nm, c in [(0, "비처방", PAL["s1"]), (1, "처방", PAL["s2"])]:
        d = data[data[key] == val]
        fig.add_trace(go.Scatter(
            x=d["baseline_hba1c"], y=d["hba1c"], mode="markers", name=nm,
            marker=dict(size=7, color=c, opacity=0.55,
                        line=dict(width=1, color=PAL["surface"])),
            hovertemplate=f"{nm}<br>기저 %{{x:.2f}}<br>26주 %{{y:.2f}}<extra></extra>"))
    fig.add_vline(x=cutoff, line=dict(color=PAL["muted"], width=2, dash="dash"),
                  annotation_text=f"기준선 {cutoff:.1f}%", annotation_position="bottom",
                  annotation_font=dict(color=PAL["muted"], size=12))
    fig.update_xaxes(title="기저 HbA1c (%)")
    fig.update_yaxes(title="26주차 HbA1c (%)")
    col.markdown(f"**{title}**")
    col.plotly_chart(style(fig), use_container_width=True)

st.subheader("주차별 평균 HbA1c")
st.caption("두 집단의 출발 수준 차이는 약효가 아니라 처방 기준선 때문입니다. "
           f"기저 HbA1c가 {cutoff:.1f}% 이상인 환자만 처방받으므로 처방군이 애초에 더 높습니다. "
           "여기서 읽어야 할 것은 높이가 아니라 기울기입니다. 보정한 추정치는 위 표에 있습니다.")
tr = A["traj"]
fig = go.Figure()
for val, nm, c in [(0, "비처방", PAL["s1"]), (1, "처방", PAL["s2"])]:
    d = tr[tr.treated == val]
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["hba1c"], mode="lines+markers+text", name=nm,
        line=dict(color=c, width=2), marker=dict(size=9, color=c,
        line=dict(width=2, color=PAL["surface"])),
        text=[f"{v:.2f}" for v in d["hba1c"]], textposition="top center",
        textfont=dict(color=PAL["ink"], size=12),
        hovertemplate=f"{nm}<br>%{{x}}주<br>%{{y:.3f}}%<extra></extra>"))
fig.update_xaxes(title="주차", tickvals=[0, 12, 26])
fig.update_yaxes(title="평균 HbA1c (%)")
st.plotly_chart(style(fig, height=320), use_container_width=True)

st.divider()
st.subheader("표와 모형 출력")
tabA, tabB = st.tabs(["교란 없음", "교란 있음"])
for tab, res in [(tabA, A), (tabB, B)]:
    with tab:
        tab.dataframe(pd.DataFrame(
            [{"방법": r[0], "추정치": round(r[1], 3), "하한": round(r[2], 3),
              "상한": round(r[3], 3), "편향": round(r[1] - true_effect, 3), "표본": r[4]}
             for r in res["rows"]]), use_container_width=True, hide_index=True)
        if "first_stage" in res:
            tab.caption(f"퍼지 RDD Wald 비율 = 축약형 점프 {res['reduced_form']:.3f} ÷ "
                        f"1단계 점프 {res['first_stage']:.3f}. 신뢰구간은 부트스트랩 "
                        f"{n_boot}회.")
        for name, txt in res["summaries"].items():
            with tab.expander(f"{name} — 모형 출력"):
                st.code(txt, language="text")

st.caption("모든 데이터는 시뮬레이션입니다. 실제 제약회사나 약물의 임상시험 결과가 아닙니다.")
