"""미관측 교란이 있을 때: 단순 회귀는 무너지고 준실험 설계는 버틴다.

analysis.py의 시뮬레이션에는 미관측 교란변수가 없다. 처방 여부가 '관측된'
기저 HbA1c만의 함수이고 결과모형도 정확히 설정되어 있으므로, 단순 회귀가
참값을 맞히는 것은 발견이 아니라 설계상 당연한 귀결이다.

이 스크립트는 그 가정을 깬다. 의사가 기저 HbA1c 기준선뿐 아니라 '차트에
기록되지 않는 중증도(U)'를 보고도 처방한다고 두면, U는 처방과 결과에
동시에 영향을 주는 미관측 교란변수가 된다. 이때 비로소 두 방법이 갈린다.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ── 0. 교란이 있는 가상 데이터 ──────────────────────────
rng = np.random.default_rng(42)
n = 8000
true_effect = -1.2
cutoff = 7.0

# U: 미관측 중증도. 연구자는 이 열을 볼 수 없다고 가정한다.
U = rng.normal(0, 1, n)
baseline_hba1c = 7.5 + 0.5 * U + rng.normal(0, 0.8, n)
above = (baseline_hba1c >= cutoff).astype(int)

# 처방은 이제 확률적(fuzzy)이다. 기준선에서 확률이 크게 점프하지만,
# 의사는 더 아픈(U가 큰) 환자에게도 더 많이 처방한다 → 교란 발생.
p_treat = np.clip(0.15 + 0.70 * above + 0.12 * U, 0.01, 0.99)
treated = rng.binomial(1, p_treat)

# U는 결과도 악화시킨다(HbA1c를 올린다).
hba1c = (baseline_hba1c
         + true_effect * treated
         + 0.8 * U
         + rng.normal(0, 0.4, n))

df = pd.DataFrame({
    "hba1c": hba1c, "treated": treated, "baseline_hba1c": baseline_hba1c,
    "above": above, "centered": baseline_hba1c - cutoff, "U": U,
})

print("=" * 64)
print("0. 교란의 크기")
print("=" * 64)
print(f"corr(처방, 미관측 중증도 U) = {np.corrcoef(treated, U)[0, 1]:.3f}")
print("→ 더 아픈 환자가 더 많이 처방받는다. 단순 회귀는 이걸 보정할 수 없다.")


print()
print("=" * 64)
print("1. 단순 회귀 (ANCOVA) — U를 볼 수 없는 현실의 분석가")
print("=" * 64)
model_naive = smf.ols("hba1c ~ treated + baseline_hba1c", data=df).fit()
print(model_naive.summary().tables[1])


print()
print("=" * 64)
print("2. 오라클 회귀 — U를 볼 수 있다면 (비교용, 현실에서는 불가능)")
print("=" * 64)
model_oracle = smf.ols("hba1c ~ treated + baseline_hba1c + U", data=df).fit()
print(model_oracle.summary().tables[1])
print("→ 참값을 회복한다. 즉 1번의 편향은 모형이 아니라 미관측 U 탓이다.")


print()
print("=" * 64)
print("3. 퍼지 회귀불연속설계 (Fuzzy RDD) — 기준선의 점프만 사용")
print("=" * 64)
bandwidth = 0.5
window = df[df["centered"].abs() <= bandwidth].copy()

# 1단계: 기준선에서 처방 확률이 얼마나 점프하는가
first_stage = smf.ols("treated ~ above * centered", data=window).fit()
# 축약형: 기준선에서 결과가 얼마나 점프하는가
reduced_form = smf.ols("hba1c ~ above * centered", data=window).fit()
# Wald 비율 = 결과 점프 / 처방 점프 = LATE
late = reduced_form.params["above"] / first_stage.params["above"]

print(f"대역폭 ±{bandwidth}  표본 수: {len(window)} / {len(df)}")
print(f"1단계 처방확률 점프 : {first_stage.params['above']:>7.3f}")
print(f"축약형 결과 점프    : {reduced_form.params['above']:>7.3f}")
print(f"Wald 비율 (LATE)    : {late:>7.3f}")

# Wald 비율의 표준오차는 OLS에서 바로 나오지 않으므로 부트스트랩으로 구한다.
boot = []
idx = np.arange(len(window))
for _ in range(1000):
    s = window.iloc[rng.choice(idx, len(idx), replace=True)]
    fs = smf.ols("treated ~ above * centered", data=s).fit()
    rf = smf.ols("hba1c ~ above * centered", data=s).fit()
    boot.append(rf.params["above"] / fs.params["above"])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"부트스트랩 95% CI   : [{lo:.3f}, {hi:.3f}]  (B=1000)")


print()
print("=" * 64)
print("4. 대역폭 민감도")
print("=" * 64)
print(f"{'대역폭':<10}{'표본':>8}{'LATE':>10}")
for bw in (0.25, 0.5, 0.75, 1.0):
    w = df[df["centered"].abs() <= bw]
    fs = smf.ols("treated ~ above * centered", data=w).fit()
    rf = smf.ols("hba1c ~ above * centered", data=w).fit()
    print(f"±{bw:<9}{len(w):>8}{rf.params['above'] / fs.params['above']:>10.3f}")


print()
print("=" * 64)
print("결과 요약")
print("=" * 64)
naive = model_naive.params["treated"]
print(f"{'방법':<26}{'처치효과':>10}{'편향':>10}")
print(f"{'단순 회귀 (U 미관측)':<26}{naive:>10.3f}{naive - true_effect:>+10.3f}")
print(f"{'오라클 회귀 (U 관측)':<26}{model_oracle.params['treated']:>10.3f}"
      f"{model_oracle.params['treated'] - true_effect:>+10.3f}")
print(f"{'퍼지 RDD':<26}{late:>10.3f}{late - true_effect:>+10.3f}")
print(f"\n(실제 설정값: {true_effect})")
print()
print(f"단순 회귀는 약효를 {abs(naive - true_effect) / abs(true_effect):.0%} 축소 보고한다.")
print("교란이 있을 때는 준실험 설계가 표본을 줄이는 값을 한다 — analysis.py와 반대 결론.")
