# Diabetes Drug Trial: Three Methods, With and Without Confounding

*[한국어 README](README.md)*

Three methods answering the same question — "how much did the new drug lower HbA1c
at week 26?" — applied in order: simple regression, a quasi-experimental design (RDD),
and a repeated-measures mixed-effects model.

Two conclusions. Without unmeasured confounding, adding complexity does not make the
estimate more accurate. With unmeasured confounding, simple regression becomes biased
and the quasi-experimental design earns the sample it costs. What matters is not the
complexity of the method but whether its identification assumptions hold.

## Data

Simulated, not real trial data. The drug is prescribed only to patients with baseline
HbA1c at or above 7.0%, mimicking a real-world prescribing threshold. The true treatment
effect is fixed at -1.2 %p, so each method can be scored on how well it recovers a known
answer.

## The three methods

1. **Simple regression (ANCOVA)** — OLS on week-26 HbA1c, adjusted for baseline
2. **Quasi-experimental design (RDD)** — local linear regression on samples near the
   7.0% prescribing threshold
3. **Mixed-effects model** — all three visits (weeks 0, 12, 26) with patient-level
   random intercepts
   - 3a. Time as linear (`treated * week`)
   - 3b. Time as categorical (`treated * C(week)`), the MMRM specification used in
     actual trials

## Running

```bash
pip install -r requirements.txt
python analysis.py              # no confounding: the three methods compared
python analysis_confounded.py   # unmeasured confounding: the conclusion reverses
```

### Interactive app

**https://diabetesdemo.streamlit.app/?lang=en**

Sliders for the true effect, sample size, prescribing threshold, bandwidth and
confounding strength, so you can watch each method's bias move in real time. The
interface is available in English and Korean; the `?lang=` parameter selects one.

```bash
streamlit run app.py
```

At default settings it reproduces the numbers in the tables above.

## Results

### Without confounding (`analysis.py`)

| Method | Estimate | n |
|---|---|---|
| Simple regression | -1.264 | 2,000 |
| RDD | -1.385 | 658 |
| Mixed-effects (linear time) | -1.295 | 2,000 |
| Mixed-effects (categorical, MMRM) | -1.243 | 2,000 |
| (true value) | -1.2 | — |

All four converge near the truth and all 95% confidence intervals cover it. The
quasi-experimental design pays for this by discarding two thirds of the sample. For the
single question "was there an effect at week 26?", simple regression was sufficient.

The mixed-effects model landed further from the truth than simple regression not because
it is complex but because **its time specification was wrong**. The simulation's actual
decay is 0.000 / 0.600 / 1.000 at weeks 0 / 12 / 26, but treating time as linear implies
0.462 at week 12. Switching to categorical time (MMRM) moves the estimate to -1.243. The
problem was specification accuracy, not complexity.

### With confounding (`analysis_confounded.py`)

The result above carries a large caveat: the world of `analysis.py` has **no unmeasured
confounding**. Treatment is a deterministic function of observed baseline HbA1c and the
outcome model is correctly specified, so simple regression recovering the truth is a
consequence of the design, not a finding.

`analysis_confounded.py` breaks that assumption. Doctors also prescribe based on a
severity term `U` that never reaches the chart, which makes assignment fuzzy at the
threshold:

| Method | Estimate | Bias | n |
|---|---|---|---|
| Simple regression (U unobserved) | -0.864 | +0.336 | 8,000 |
| Oracle regression (U observed) | -1.215 | -0.015 | 8,000 |
| Fuzzy RDD | -1.224 | -0.024 | 2,934 |
| (true value) | -1.2 | — | — |

Simple regression **understates the drug's benefit by 28%**. The oracle regression, which
includes `U`, recovers the truth — confirming the bias comes from unmeasured confounding
rather than from the model. Fuzzy RDD uses only the jump at the threshold, so `U` does not
touch it (bootstrap 95% CI [-1.384, -1.060], B=1000).

### So

The quasi-experimental design is not free — it cuts the sample from 8,000 to 2,934 — but
**when confounding is present it is worth that cost.** Whether a simple method suffices
depends on whether the identification assumptions hold, not on how complex the method is.

## Notes

- All data is simulated with `numpy` seed 42. These are not results from any real
  pharmaceutical company or drug.
- The seed is fixed, so re-running reproduces the same numbers.
- The fuzzy RDD in `analysis_confounded.py` estimates the LATE via the Wald ratio
  (reduced-form jump ÷ first-stage jump). Its standard error does not come directly from
  OLS, so it is bootstrapped. A bandwidth sensitivity table (±0.25 to ±1.0) is printed in
  section 4 of the script.

## License

MIT. See [LICENSE](LICENSE).
