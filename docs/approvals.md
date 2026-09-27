# Pre-Execution Approvals — EDA Plan

Ivania Zavidich-Fearnow · IS 6812 · Approves `docs/eda_plan.md` for execution

These are the five decisions `eda_plan.md` requested before code runs, plus one correction. Each is a
judgment call about my business problem, not something the data can settle, so each states the rule and
the reason.

**Standing rule for all treatments:** never modify the source CSVs. All recoding happens in a derived
DataFrame built at the top of the notebook, so every transformation is visible and reversible.

---

## 1. Anomaly treatments (S4)

| Finding | Treatment | Reason |
|---|---|---|
| `DAYS_EMPLOYED = 365243` (55,374 train / 9,274 test) | Set to `NaN`; add `FLAG_NOT_EMPLOYED = 1`. **Do not drop the rows.** | It is a sentinel for "not employed," not a measurement. The rows are 18% of the data and default at 5.40% vs 8.66% — a lower-risk group. Dropping them would bias the sample toward working applicants and discard a real signal. |
| `CODE_GENDER = 'XNA'` (4 train rows) | Drop the 4 rows from train. | 4 of 307,511. Too few to model, too few to matter. State the drop rather than silently filtering. |
| `REGION_RATING_CLIENT_W_CITY = -1` (1 **test** row) | Recode to the modal value; **do not drop.** | Test rows must be scored. A row cannot be dropped from a population I am required to produce a prediction for. This is the general rule: train rows may be dropped, test rows may only be repaired. |
| `AMT_INCOME_TOTAL` max 117,000,000 (99.9th pct = 900,000) | Keep raw values in all tables. Cap at the 99.9th percentile **for plotting only**, and label the axis as capped. | Deleting outliers changes the base rate. The extreme value may be a data-entry error or a genuinely wealthy applicant, and this data cannot distinguish them. Winsorizing the plot fixes readability without altering the analysis. |
| `OBS_*/DEF_*_CNT_SOCIAL_CIRCLE` max 344 | Keep, flag as suspect in the anomaly register, exclude from headline findings. | 344 observable social-circle contacts is not credible, but there is no documented domain to call it impossible. `feature_report.md` already flags this family for privacy and explainability risk, so it should not carry a conclusion. |

---

## 2. Thresholds (S6, Q2b, and binning)

**Redundancy grouping:** group numeric columns at |Spearman ρ| ≥ 0.90. Within each group keep the
member with the highest non-missing coverage; break ties on interpretability, preferring the column an
underwriter could explain. Record the full group membership — do not silently discard members.

*Reason:* 0.90 is high enough that grouped columns are measuring one construct rather than two related
ones. The `_AVG` / `_MODE` / `_MEDI` triplets and `AMT_CREDIT` / `AMT_GOODS_PRICE` should fall out
automatically; if they don't, the threshold is wrong.

**Minimum group size: 1,000 applicants** for any bin or category level reported as a finding.

*Reason:* at an 8.07% base rate, 1,000 applicants yields roughly 81 expected events and a standard
error near 0.9 percentage points, so a rate is known to about ±1.7 points at 95% confidence. Below
that, the interval is too wide to distinguish a segment from the overall population. Levels under 1,000
are pooled into "Other" and never reported standalone.

**Missingness exclusion: > 60% missing** removes a column from the Q1 separation ranking.

*Reason:* above 60%, a decile plot is built mostly on absence. The column is still tested in Q2a for
missingness-as-signal — exclusion from the ranking is not exclusion from the analysis. Note this keeps
`EXT_SOURCE_1` (56% missing) in the ranking, which is intentional: it is the variable the screen in Q2
exists to adjudicate.

---

## 3. "Strong enough to stand alone" (Q1)

A variable stands alone only if it clears **both**:

1. **Univariate ROC AUC ≥ 0.70** on a held-out fold, and
2. **Worst-decile to best-decile default-rate ratio ≥ 2.0×** — the riskiest tenth of applicants defaults
   at least twice as often as the safest tenth.

*Reason:* AUC 0.50 is a coin flip. 0.70 is the conventional floor for a usable standalone score, and
the decile ratio is the same claim in language a credit committee can act on. Requiring both prevents
a variable from passing on a statistic nobody outside the notebook can interpret.

**Pre-registered expectation:** I expect **no single variable to clear 0.70**, and I expect the
`EXT_SOURCE` columns to come closest. If that holds, the finding is that this problem has no
single-variable solution — which is precisely the evidence that justifies a multivariable model in the
next stage. Recording the prediction now means the result is a test rather than a rationalization.

---

## 4. Verifiability rubric and screen thresholds (Q2)

`eda_plan.md` is right that these CSVs contain no verification status. This is therefore a documented
qualitative rubric applied from the data dictionary and `feature_report.md`, not a measurement.

**Tier A — document-backed.** Verifiable from a document the applicant supplies or the lender pulls:
income, employment, age, education, family status, housing type, contract amounts, car ownership.

**Tier B — self-reported or structural.** Cheap to collect, weakly verifiable or not verifiable:
contact flags, phone availability, region ratings, process timing.

**Tier C — opaque.** Cannot be verified, explained, or reproduced: `EXT_SOURCE_1/2/3`, the
social-circle counts. `feature_report.md` establishes their provenance is undocumented.

**Dispositions:**

- **Pass** — Tier A, coverage ≥ 80%, and meaningful signal in Q1.
- **Review** — Tier B, or Tier A with coverage 50–80%.
- **Fail** — Tier C, or any tier below 50% coverage.

*Reason, and the point of the exercise:* this rubric fails the `EXT_SOURCE` variables, which are
probably the strongest predictors in the dataset. That is the intended behavior, because my BPS commits
to variables that are available and easy to verify, and because an adverse-action notice cannot say
"your undocumented external score was low." **The notebook must still report their predictive strength**
— the cost of excluding them has to be visible, or the recommendation isn't honest.

Predictive strength is kept separate from fair-lending permissibility. A variable can pass this screen
and still require governance review; flag those rather than resolving them here.

---

## 5. Material stability threat (Q3)

Two tiers, both requiring the segment to have ≥ 1,000 applicants and a 95% confidence interval that
excludes the overall 8.07% rate:

- **Note it** — segment default rate differs from the overall rate by ≥ 1.25× relative (i.e., below
  6.5% or above 10.1%).
- **Threatens stability** — either the relative difference is ≥ 2.0× (below 4.0% or above 16.1%), **or**
  a predictor's direction reverses between segments.

*Reason:* a direction reversal is the serious case. A variable that signals higher risk in one segment
and lower risk in another cannot be deployed under a single rule, and that is exactly the failure mode
my BPS worries about when it questions whether predictors transfer across populations. A large but
consistent difference is a segmentation opportunity; a reversal is a defect.

---

## 6. Correction to the plan

`eda_plan.md` closes by stating that the approval-threshold trade-off question "is absent from the
submitted question list." **It is not absent.** It is in `docs/questions.md` under *"Deferred to the
modeling stage"* as item D1, with the reason recorded: answering it requires building a composite score
from the `EXT_SOURCE` columns, and a composite fit and evaluated on the same data is a model — one that
would produce a falsely optimistic trade-off curve presented as an exploratory finding.

Please record it as **deferred with a stated reason**, not as an omission. The notebook should close by
naming what a threshold analysis will require: out-of-sample scores, a held-out validation split, and a
stated cost ratio between a missed default and a declined good applicant.

---

## Approved for execution

With the six items above resolved, `docs/eda_plan.md` is approved. Proceed to Chat 4.
