"""교란 유무에 따라 분석 방법의 성능이 어떻게 갈리는지 보여주는 대화형 데모.

Interactive demo: how unmeasured confounding decides whether a simple method suffices.
UI는 한국어/영어를 지원한다. ?lang=en 으로 바로 열 수 있다.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="Diabetes Drug Trial: Methods", layout="wide")

# ── 문자열 ────────────────────────────────────────────
# 방법 이름은 캐시 함수가 키로 돌려주고, 화면에 그릴 때 번역한다.
METHOD = {
    "ols":       {"ko": "단순 회귀",            "en": "Simple regression"},
    "rdd":       {"ko": "RDD",                  "en": "RDD"},
    "mixed_lin": {"ko": "혼합효과 (선형 시간)",  "en": "Mixed-effects (linear time)"},
    "mmrm":      {"ko": "혼합효과 (MMRM)",       "en": "Mixed-effects (MMRM)"},
    "naive":     {"ko": "단순 회귀 (U 미관측)",  "en": "Simple regression (U unobserved)"},
    "oracle":    {"ko": "오라클 회귀 (U 관측)",  "en": "Oracle regression (U observed)"},
    "fuzzy":     {"ko": "퍼지 RDD",              "en": "Fuzzy RDD"},
    "fs":        {"ko": "퍼지 RDD 1단계",        "en": "Fuzzy RDD first stage"},
    "rf":        {"ko": "퍼지 RDD 축약형",       "en": "Fuzzy RDD reduced form"},
}

T = {
    "ko": {
        "title": "교란이 있을 때와 없을 때, 방법은 어떻게 갈리는가",
        "sub": "같은 질문에 네 가지 방법으로 답합니다. 참값을 미리 고정했으므로 "
               "각 방법이 그 값을 얼마나 잘 회복하는지 직접 비교할 수 있습니다.",
        "sim_header": "시뮬레이션 설정", "conf_header": "교란 설정",
        "true_effect": "참 처치효과 (%p)", "n_clean": "표본 수 (교란 없음)",
        "cutoff": "처방 기준선 (HbA1c %)", "bandwidth": "RDD 대역폭",
        "n_conf": "표본 수 (교란 있음)",
        "n_conf_help": "퍼지 RDD는 Wald 비율이라 대역폭 안 표본이 얇으면 추정이 "
                       "불안정합니다. 8000 이상을 권합니다.",
        "u_note": "U는 차트에 기록되지 않는 중증도입니다.",
        "gamma": "U → 처방 (교란 강도)", "delta": "U → 결과",
        "n_boot": "부트스트랩 반복", "seed": "난수 시드",
        "fitting": "모형 적합 중...",
        "panel_clean": "교란 없음", "panel_conf": "미관측 교란 있음",
        "cap_clean": "처방이 관측된 기저치만의 함수. 단순 회귀가 맞는 것은 설계상 당연합니다.",
        "cap_conf": "U가 처방과 결과에 동시에 작용합니다. 표본 {n:,}. corr(처방, U) = {c:.3f}",
        "bias": "편향", "sample": "표본",
        "forest_clean": "교란 없음", "forest_conf": "교란 있음",
        "true_label": "참값", "xaxis_effect": "26주차 처치효과 (%p)",
        "win": "교란이 있으면 단순 회귀 편향({a:.3f})이 퍼지 RDD 편향({b:.3f})보다 큽니다. "
               "준실험 설계가 표본을 줄이는 비용을 정당화합니다.",
        "tie": "현재 설정에서는 두 방법의 편향 차이가 크지 않습니다 "
               "(단순 회귀 {a:.3f}, 퍼지 RDD {b:.3f}). 교란 강도 슬라이더를 올리면 차이가 드러납니다.",
        "disc": "기준선에서의 불연속",
        "sc_clean": "교란 없음: 처방이 결정적", "sc_conf": "교란 있음: 처방이 확률적",
        "untreated": "비처방", "treated": "처방",
        "cut_label": "기준선 {v:.1f}%",
        "x_base": "기저 HbA1c (%)", "y_w26": "26주차 HbA1c (%)",
        "traj": "주차별 평균 HbA1c",
        "traj_note": "두 집단의 출발 수준 차이는 약효가 아니라 처방 기준선 때문입니다. "
                     "기저 HbA1c가 {v:.1f}% 이상인 환자만 처방받으므로 처방군이 애초에 더 높습니다. "
                     "여기서 읽어야 할 것은 높이가 아니라 기울기입니다. 보정한 추정치는 위 표에 있습니다.",
        "x_week": "주차", "y_mean": "평균 HbA1c (%)",
        "tables": "표와 모형 출력",
        "col_method": "방법", "col_est": "추정치", "col_lo": "하한", "col_hi": "상한",
        "col_bias": "편향", "col_n": "표본",
        "wald": "퍼지 RDD Wald 비율 = 축약형 점프 {rf:.3f} ÷ 1단계 점프 {fs:.3f}. "
                "신뢰구간은 부트스트랩 {b}회.",
        "model_out": "모형 출력",
        "footer": "모든 데이터는 시뮬레이션입니다. 실제 제약회사나 약물의 임상시험 결과가 아닙니다.",
        "lang_label": "언어 / Language",
    },
    "en": {
        "title": "What decides whether a simple method suffices",
        "sub": "Four methods answering the same question. The true effect is fixed in "
               "advance, so each method can be scored on how well it recovers a known answer.",
        "sim_header": "Simulation", "conf_header": "Confounding",
        "true_effect": "True treatment effect (%p)", "n_clean": "Sample size (no confounding)",
        "cutoff": "Prescribing threshold (HbA1c %)", "bandwidth": "RDD bandwidth",
        "n_conf": "Sample size (confounded)",
        "n_conf_help": "Fuzzy RDD is a Wald ratio, so a thin bandwidth window makes it "
                       "unstable. 8000 or more is recommended.",
        "u_note": "U is a severity term that never reaches the chart.",
        "gamma": "U → prescribing (confounding strength)", "delta": "U → outcome",
        "n_boot": "Bootstrap replications", "seed": "Random seed",
        "fitting": "Fitting models...",
        "panel_clean": "No confounding", "panel_conf": "Unmeasured confounding",
        "cap_clean": "Treatment is a deterministic function of observed baseline. Simple "
                     "regression being right follows from the design.",
        "cap_conf": "U drives both prescribing and outcome. n = {n:,}. corr(treated, U) = {c:.3f}",
        "bias": "bias", "sample": "n",
        "forest_clean": "No confounding", "forest_conf": "With confounding",
        "true_label": "true", "xaxis_effect": "Treatment effect at week 26 (%p)",
        "win": "With confounding, simple regression's bias ({a:.3f}) exceeds fuzzy RDD's "
               "({b:.3f}). The quasi-experimental design earns the sample it costs.",
        "tie": "At these settings the two biases are close (simple regression {a:.3f}, "
               "fuzzy RDD {b:.3f}). Raise the confounding strength slider to separate them.",
        "disc": "Discontinuity at the threshold",
        "sc_clean": "No confounding: assignment is deterministic",
        "sc_conf": "With confounding: assignment is probabilistic",
        "untreated": "Untreated", "treated": "Treated",
        "cut_label": "threshold {v:.1f}%",
        "x_base": "Baseline HbA1c (%)", "y_w26": "Week-26 HbA1c (%)",
        "traj": "Mean HbA1c by week",
        "traj_note": "The gap in starting level is the prescribing threshold, not a drug "
                     "effect. Only patients at or above {v:.1f}% baseline HbA1c are treated, "
                     "so that group starts higher. Read the slope, not the height. Adjusted "
                     "estimates are in the table above.",
        "x_week": "Week", "y_mean": "Mean HbA1c (%)",
        "tables": "Tables and model output",
        "col_method": "Method", "col_est": "Estimate", "col_lo": "Lower", "col_hi": "Upper",
        "col_bias": "Bias", "col_n": "n",
        "wald": "Fuzzy RDD Wald ratio = reduced-form jump {rf:.3f} ÷ first-stage jump "
                "{fs:.3f}. Confidence interval bootstrapped over {b} replications.",
        "model_out": "model output",
        "footer": "All data is simulated. These are not results from any real "
                  "pharmaceutical company or drug.",
        "lang_label": "언어 / Language",
    },
}

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

    return {
        "rows": [("ols", *ols, len(w26)), ("rdd", *rdd, len(win)),
                 ("mixed_lin", *lin, len(w26)), ("mmrm", *mmrm, len(w26))],
        "scatter": w26.sample(min(700, len(w26)), random_state=0),
        "traj": long.groupby(["week", "treated"])["hba1c"].mean().reset_index(),
        "summaries": {"ols": m_ols.summary().tables[1].as_text(),
                      "rdd": m_rdd.summary().tables[1].as_text(),
                      "mixed_lin": str(m_lin.summary().tables[1]),
                      "mmrm": str(m_mmrm.summary().tables[1])},
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
        "rows": [("naive", *naive, len(df)), ("oracle", *orac, len(df)),
                 ("fuzzy", late, lo, hi, len(win))],
        "corr": float(np.corrcoef(treated, U)[0, 1]),
        "first_stage": float(fs.params["above"]), "reduced_form": float(rf.params["above"]),
        "scatter": df.sample(min(700, len(df)), random_state=0),
        "summaries": {"naive": m_naive.summary().tables[1].as_text(),
                      "oracle": m_orac.summary().tables[1].as_text(),
                      "fs": fs.summary().tables[1].as_text(),
                      "rf": rf.summary().tables[1].as_text()},
    }


def forest(rows, true_effect, title, t, lang):
    """추정치와 신뢰구간. 값은 오른쪽 끝에 한 열로 모아 찍는다(대비 완화 규칙)."""
    labels = [METHOD[r[0]][lang] for r in rows][::-1]
    est = [r[1] for r in rows][::-1]
    lo = [r[1] - r[2] for r in rows][::-1]
    hi = [r[3] - r[1] for r in rows][::-1]
    colors = [PAL["s1"], PAL["s2"], PAL["s3"], PAL["s1"]][:len(rows)][::-1]

    fig = go.Figure()
    fig.add_vline(x=true_effect, line=dict(color=PAL["muted"], width=2, dash="dash"),
                  annotation_text=f"{t['true_label']} {true_effect:.2f}",
                  annotation_position="top", annotation_yshift=6,
                  annotation_font=dict(color=PAL["muted"], size=12))
    fig.add_trace(go.Scatter(
        x=est, y=labels, mode="markers", marker=dict(size=13, color=colors,
        line=dict(width=2, color=PAL["surface"])),
        error_x=dict(type="data", array=hi, arrayminus=lo, color=PAL["muted"],
                     width=6, thickness=2),
        showlegend=False, hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))

    span_lo = min(min(r[2] for r in rows), true_effect)
    span_hi = max(max(r[3] for r in rows), true_effect)
    pad = max(0.12, (span_hi - span_lo) * 0.10)
    fig.add_trace(go.Scatter(
        x=[span_hi + pad * 1.35] * len(est), y=labels, mode="text",
        text=[f"{v:.3f}" for v in est], textposition="middle right",
        textfont=dict(color=PAL["ink"], size=13), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=title)
    fig.update_xaxes(title=t["xaxis_effect"], range=[span_lo - pad, span_hi + pad * 3.2])
    fig.update_yaxes(title=None)
    return style(fig, height=300)


# ══ 화면 ══════════════════════════════════════════════
_qp = st.query_params.get("lang", "ko")
_default = 1 if str(_qp).lower().startswith("en") else 0
with st.sidebar:
    lang = st.radio(T["ko"]["lang_label"], ["ko", "en"], index=_default, horizontal=True,
                    format_func=lambda v: "한국어" if v == "ko" else "English")
st.query_params["lang"] = lang
t = T[lang]

st.title(t["title"])
st.caption(t["sub"])

with st.sidebar:
    st.header(t["sim_header"])
    true_effect = st.slider(t["true_effect"], -2.5, 0.0, -1.2, 0.1)
    n = st.select_slider(t["n_clean"], [500, 1000, 2000, 4000, 8000], 2000)
    cutoff = st.slider(t["cutoff"], 6.5, 8.0, 7.0, 0.1)
    bandwidth = st.slider(t["bandwidth"], 0.25, 1.5, 0.5, 0.05)
    st.divider()
    st.header(t["conf_header"])
    st.caption(t["u_note"])
    n_conf = st.select_slider(t["n_conf"], [2000, 4000, 8000, 16000], 8000,
                              help=t["n_conf_help"])
    gamma = st.slider(t["gamma"], 0.0, 0.40, 0.12, 0.01)
    delta = st.slider(t["delta"], 0.0, 1.5, 0.8, 0.1)
    n_boot = st.select_slider(t["n_boot"], [100, 200, 300, 500, 1000], 300)
    seed = st.number_input(t["seed"], 0, 9999, 42)

with st.spinner(t["fitting"]):
    A = run_clean(n, true_effect, cutoff, bandwidth, seed)
    B = run_confounded(n_conf, true_effect, cutoff, bandwidth, gamma, delta, n_boot, seed)

left, right = st.columns(2, gap="large")
with left:
    st.subheader(t["panel_clean"])
    st.caption(t["cap_clean"])
    for key, e, _, _, k in A["rows"]:
        st.metric(METHOD[key][lang], f"{e:.3f}", f"{e - true_effect:+.3f} {t['bias']}",
                  delta_color="off", help=f"{t['sample']} {k:,}")
with right:
    st.subheader(t["panel_conf"])
    st.caption(t["cap_conf"].format(n=n_conf, c=B["corr"]))
    for key, e, _, _, k in B["rows"]:
        st.metric(METHOD[key][lang], f"{e:.3f}", f"{e - true_effect:+.3f} {t['bias']}",
                  delta_color="off", help=f"{t['sample']} {k:,}")

st.divider()
f1, f2 = st.columns(2, gap="large")
f1.plotly_chart(forest(A["rows"], true_effect, t["forest_clean"], t, lang),
                use_container_width=True)
f2.plotly_chart(forest(B["rows"], true_effect, t["forest_conf"], t, lang),
                use_container_width=True)

naive_bias = abs(B["rows"][0][1] - true_effect)
rdd_bias = abs(B["rows"][2][1] - true_effect)
if naive_bias > rdd_bias * 1.5:
    st.success(t["win"].format(a=naive_bias, b=rdd_bias))
else:
    st.info(t["tie"].format(a=naive_bias, b=rdd_bias))

st.divider()
st.subheader(t["disc"])
s1, s2 = st.columns(2, gap="large")
for col, data, title in [(s1, A["scatter"], t["sc_clean"]), (s2, B["scatter"], t["sc_conf"])]:
    fig = go.Figure()
    for val, nm, c in [(0, t["untreated"], PAL["s1"]), (1, t["treated"], PAL["s2"])]:
        d = data[data["treated"] == val]
        fig.add_trace(go.Scatter(
            x=d["baseline_hba1c"], y=d["hba1c"], mode="markers", name=nm,
            marker=dict(size=7, color=c, opacity=0.55,
                        line=dict(width=1, color=PAL["surface"])),
            hovertemplate=f"{nm}<br>%{{x:.2f}} → %{{y:.2f}}<extra></extra>"))
    fig.add_vline(x=cutoff, line=dict(color=PAL["muted"], width=2, dash="dash"),
                  annotation_text=t["cut_label"].format(v=cutoff),
                  annotation_position="bottom",
                  annotation_font=dict(color=PAL["muted"], size=12))
    fig.update_xaxes(title=t["x_base"])
    fig.update_yaxes(title=t["y_w26"])
    col.markdown(f"**{title}**")
    col.plotly_chart(style(fig), use_container_width=True)

st.subheader(t["traj"])
st.caption(t["traj_note"].format(v=cutoff))
tr = A["traj"]
fig = go.Figure()
for val, nm, c in [(0, t["untreated"], PAL["s1"]), (1, t["treated"], PAL["s2"])]:
    d = tr[tr.treated == val]
    fig.add_trace(go.Scatter(
        x=d["week"], y=d["hba1c"], mode="lines+markers+text", name=nm,
        line=dict(color=c, width=2),
        marker=dict(size=9, color=c, line=dict(width=2, color=PAL["surface"])),
        text=[f"{v:.2f}" for v in d["hba1c"]], textposition="top center",
        textfont=dict(color=PAL["ink"], size=12),
        hovertemplate=f"{nm}<br>%{{x}}<br>%{{y:.3f}}%<extra></extra>"))
fig.update_xaxes(title=t["x_week"], tickvals=[0, 12, 26])
fig.update_yaxes(title=t["y_mean"])
st.plotly_chart(style(fig, height=320), use_container_width=True)

st.divider()
st.subheader(t["tables"])
tabA, tabB = st.tabs([t["panel_clean"], t["panel_conf"]])
for tab, res in [(tabA, A), (tabB, B)]:
    with tab:
        tab.dataframe(pd.DataFrame(
            [{t["col_method"]: METHOD[r[0]][lang], t["col_est"]: round(r[1], 3),
              t["col_lo"]: round(r[2], 3), t["col_hi"]: round(r[3], 3),
              t["col_bias"]: round(r[1] - true_effect, 3), t["col_n"]: r[4]}
             for r in res["rows"]]), use_container_width=True, hide_index=True)
        if "first_stage" in res:
            tab.caption(t["wald"].format(rf=res["reduced_form"], fs=res["first_stage"],
                                         b=n_boot))
        for key, txt in res["summaries"].items():
            with tab.expander(f"{METHOD[key][lang]} — {t['model_out']}"):
                st.code(txt, language="text")

st.caption(t["footer"])
