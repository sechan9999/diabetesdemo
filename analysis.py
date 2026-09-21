import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ── 0. 가상 데이터 생성 ──────────────────────────────
rng = np.random.default_rng(42)
n = 2000

baseline_hba1c = rng.normal(7.5, 1.0, n)
treated = (baseline_hba1c >= 7.0).astype(int)  # 기저 HbA1c 7.0% 기준 처방
patient_id = np.arange(n)
age = rng.normal(55, 10, n)
bmi = rng.normal(31, 4, n)

true_effect = -1.2
patient_random_effect = rng.normal(0, 0.5, n)

def make_visit(week, decay):
    noise = rng.normal(0, 0.4, n)
    hba1c = (baseline_hba1c
             + decay * (true_effect * treated)
             + patient_random_effect
             + noise)
    return pd.DataFrame({
        "patient_id": patient_id, "week": week, "hba1c": hba1c,
        "baseline_hba1c": baseline_hba1c, "treated": treated,
        "age": age, "bmi": bmi,
    })

df_long = pd.concat([make_visit(0, 0.0), make_visit(12, 0.6), make_visit(26, 1.0)],
                     ignore_index=True)
df_w26 = df_long[df_long.week == 26].copy()


print("=" * 60)
print("1. 단순 회귀 (ANCOVA)")
print("=" * 60)
model_simple = smf.ols("hba1c ~ treated + baseline_hba1c", data=df_w26).fit()
print(model_simple.summary().tables[1])


print()
print("=" * 60)
print("2. 준실험 설계: 회귀불연속설계(RDD)")
print("=" * 60)
cutoff, bandwidth = 7.0, 0.5
df_w26["centered"] = df_w26["baseline_hba1c"] - cutoff
window = df_w26[df_w26["centered"].abs() <= bandwidth].copy()

model_rdd = smf.ols("hba1c ~ treated * centered", data=window).fit()
print(f"대역폭 내 표본 수: {len(window)} / {len(df_w26)}")
print(model_rdd.summary().tables[1])


print()
print("=" * 60)
print("3a. 복잡한 통계모형: 혼합효과모형 (시간을 선형으로)")
print("=" * 60)
model_mixed = smf.mixedlm(
    "hba1c ~ treated * week + baseline_hba1c",
    data=df_long,
    groups=df_long["patient_id"],
).fit()
print(model_mixed.summary().tables[1])

effect_w26 = (model_mixed.params["treated"]
              + model_mixed.params["treated:week"] * 26)
print(f"\n26주차 처치효과: {effect_w26:.3f}")
print("주의: 이 설정은 효과가 주차에 비례한다고 가정한다.")
print(f"  실제 감쇠 패턴   : 0주 0.000 / 12주 {0.6:.3f} / 26주 1.000")
print(f"  선형 가정의 함의 : 0주 0.000 / 12주 {12/26:.3f} / 26주 1.000  ← 불일치")


print()
print("=" * 60)
print("3b. 혼합효과모형: 시간을 범주형으로 (MMRM)")
print("=" * 60)
# 실제 임상시험에서 쓰는 MMRM 설정. 두 가지를 고친다.
#   (1) C(week)으로 시간 선형성 가정을 버린다.
#   (2) 기저치를 공변량으로 쓰므로 0주차는 반응변수에서 제외한다.
df_post = df_long[df_long.week > 0].copy()
model_mmrm = smf.mixedlm(
    "hba1c ~ treated * C(week) + baseline_hba1c",
    data=df_post,
    groups=df_post["patient_id"],
).fit()
print(model_mmrm.summary().tables[1])

effect_mmrm = (model_mmrm.params["treated"]
               + model_mmrm.params["treated:C(week)[T.26]"])
print(f"\nMMRM의 26주차 처치효과: {effect_mmrm:.3f}")


print()
print("=" * 60)
print("결과 요약")
print("=" * 60)
print(f"{'방법':<26}{'처치효과':>10}{'표본':>10}")
print(f"{'단순 회귀':<26}{model_simple.params['treated']:>10.3f}{len(df_w26):>10}")
print(f"{'RDD':<26}{model_rdd.params['treated']:>10.3f}{len(window):>10}")
print(f"{'혼합효과 (선형 시간)':<26}{effect_w26:>10.3f}{len(df_w26):>10}")
print(f"{'혼합효과 (범주형, MMRM)':<26}{effect_mmrm:>10.3f}{len(df_w26):>10}")
print(f"\n(실제 설정값: {true_effect})")
print()
print("혼합효과모형이 단순 회귀보다 나빴던 것은 복잡해서가 아니라")
print("시간 설정이 틀렸기 때문이다. 고치면(3b) 참값에 더 가까워진다.")
print()
print("단, 이 시뮬레이션에는 미관측 교란이 없다 — 처방이 '관측된' 기저치만의")
print("함수이므로 단순 회귀가 맞는 것은 설계상 당연하다.")
print("교란을 넣으면 결론이 뒤집힌다: analysis_confounded.py 참고.")
