# EDA Plan — Home Credit Default Risk

## Scope and execution status

This is a plan only. No analysis code has been run. The decision rules in this plan were approved in `docs/approvals.md`; execution remains a separate next stage.

The EDA will use `application_train.csv` and `application_test.csv`. This matches the scope in `docs/eda_workflow_guide.md`: one row represents one current loan application, `application_train` supplies the outcome, and `application_test` is the held-out scoring population. The supplementary history tables are out of scope for this EDA. The column dictionary, `data_map.md`, and `feature_report.md` may be used as documentation, not as analytic observations.

`TARGET = 1` means repayment difficulty and `TARGET = 0` means no repayment difficulty. IDs will be used only for integrity checks, never as predictors.

Status labels used below:

- **Answerable:** the in-scope data can answer the question as written after its decision rule is specified.
- **Partially answerable:** the data support a narrower claim than the wording suggests.
- **Cannot answer:** the required observations are absent.
- **Description/label:** the item is not itself a question and produces no standalone finding.

## Audit against the six standing checks

| Standing check | Covered by the submitted questions? | Plan decision |
|---|---|---|
| Target distribution | No | Add S2 and mark it `[added]`. |
| Missingness, including its relationship to `TARGET` | Yes, by Q2a | Reorder it into the standing checks as S6. |
| Impossible values | No | Add S4 and mark it `[added]`. |
| Keys, duplicates, and table grain | No | Add S1 and mark it `[added]`. |
| Temporal direction / known at application time | No | Add S5 and mark it `[added]`. |
| Train/test consistency in columns and category levels | No | Add S3 and mark it `[added]`. |

The heading **“The Standing Checks”** and its six bullets in `docs/questions.md` are requirements/descriptions, not questions. Merely listing the checks does not specify what will be tested or what evidence will answer them, so the uncovered checks are converted to questions below.

## Ordered question plan

The order below establishes data grain and comparability before any relationship with the outcome is calculated. Data-quality decisions then precede redundancy, univariate separation, screening, and segment comparisons.

### S1. `[added]` Is each in-scope table at the expected one-row-per-current-application grain, with a unique key and no duplicate rows?

- **Status:** Answerable.
- **Method:** For train and test, report row and column counts; test `SK_ID_CURR` for nulls and uniqueness; count fully duplicated rows; check whether train and test ID sets overlap. Confirm that one row is one current application—not necessarily one persistent customer or one originated loan.
- **Tables needed:** `application_train`, `application_test`.
- **Output:** A two-row integrity table with grain, dimensions, null keys, distinct keys, duplicate-key rows, full-row duplicates, and train/test ID overlap; followed by one sentence confirming or rejecting the expected grain.
- **Dependency created:** Every later applicant-level rate or comparison relies on this result.

### S2. `[added]` What is the distribution of repayment difficulty in the training data?

- **Status:** Answerable for train; `application_test` has no observed outcome by design.
- **Method:** Count and calculate the proportion of `TARGET = 0`, `TARGET = 1`, and missing/invalid target values. State the majority-class accuracy only as an imbalance baseline, not as a useful model.
- **Tables needed:** `application_train` (`TARGET`).
- **Output:** A count/percentage table and a labeled bar chart for the two target classes, plus a short baseline statement.
- **Dependency created:** Supplies the base rate against which all later default-rate differences and relative changes are interpreted.

### S3. `[added]` Are train and test consistent in columns, data types, and categorical levels?

- **Status:** Answerable for observed schema and levels; it cannot establish whether future production data will remain consistent.
- **Method:** Compare column sets after allowing the expected train-only `TARGET`; compare corresponding data types; for each categorical field, identify levels unique to train or test and their counts/shares; note columns that are entirely missing or constant in either split. Do not use `TARGET` to “correct” these differences.
- **Tables needed:** `application_train`, `application_test`.
- **Output:** (1) schema-difference table, (2) dtype-mismatch table, and (3) category-level mismatch table with field, level, split, count, and share. A compact summary should distinguish expected from actionable differences.
- **Dependency created:** Establishes which variables and encodings can be compared and later deployed.

### S4. `[added]` Which values are impossible, undocumented sentinels, or implausible enough to distort later results?

- **Status:** Partially answerable. Dictionary contradictions and obvious sentinels are testable; “implausible” values require a documented business rule and cannot be declared errors from rarity alone.
- **Method:** Apply dictionary domains and sign/range rules; summarize numeric minima, maxima, and extreme quantiles; inspect rare categorical levels; explicitly test known concerns such as `DAYS_EMPLOYED = 365243`, positive `DAYS_*` values, `CODE_GENDER = 'XNA'`, extreme `AMT_INCOME_TOTAL`, and extreme social-circle counts. Classify each finding as impossible, documented/undocumented sentinel, merely extreme, or unresolved. Apply the approved treatments only in derived DataFrames: set `DAYS_EMPLOYED = 365243` to `NaN`, add `FLAG_NOT_EMPLOYED`, and retain those rows; drop the four `CODE_GENDER = 'XNA'` rows from train; replace the one test value `REGION_RATING_CLIENT_W_CITY = -1` with the training mode and never drop test rows; retain raw `AMT_INCOME_TOTAL` values in analysis tables but cap plots at the training 99.9th percentile with a labeled capped axis; retain social-circle extremes, flag them as suspect, and exclude them from headline claims. Never modify the source CSVs.
- **Tables needed:** `application_train`, `application_test`, and `HomeCredit_columns_description.csv` as metadata.
- **Output:** An anomaly register with column, rule, offending value/range, train count/share, test count/share, classification, and applied treatment; optional focused plots for high-impact numeric anomalies. The transformation log will state the train-row count before and after the approved `XNA` removal and confirm unchanged test-row count.
- **Dependency created:** Produces the cleaned analysis representation used by all later calculations.

### S5. `[added]` Were all proposed predictors known at the current application decision time?

- **Status:** Partially answerable. The dictionary and table design support an application-time interpretation, but the dataset does not contain full source-system timestamps or data-lineage evidence for every field—especially the opaque `EXT_SOURCE_*` scores.
- **Method:** Classify each candidate field as application-time input, pre-application history, outcome, identifier, or timing/provenance unclear. Check the sign convention of relative-time variables after treating sentinels identified in S4. Exclude `TARGET` and IDs from predictors. Flag any field whose timing cannot be substantiated rather than assuming it is safe.
- **Tables needed:** `application_train`, `application_test`, and `HomeCredit_columns_description.csv`; consult the timing limitations in `feature_report.md`.
- **Output:** A feature-timing inventory with field/family, timing class, evidence, leakage risk, and keep/exclude/review decision.
- **Dependency created:** Defines the leakage-safe candidate set for subsequent questions.

### S6 / Q2a. Is missingness a defect to repair, or a signal in itself?

- **Status:** Partially answerable. The data can show whether missingness is associated with `TARGET`, but cannot reveal why a value is missing or prove that the mechanism will persist in deployment.
- **Method:** For every candidate field, calculate missing count/share in train and test and their percentage-point difference. In train, compare the repayment-difficulty rate for missing versus present rows, including group sizes, absolute difference, relative lift, and uncertainty intervals. Treat sentinel values identified in S4 separately from ordinary nulls. A field with more than 60% missing remains in Q2a but is excluded from the Q1 separation ranking. Avoid overinterpreting tiny groups and multiple unadjusted significance tests.
- **Tables needed:** `application_train`, `application_test`; `TARGET` is used only for the train association.
- **Output:** (1) ranked missingness table, (2) train-versus-test missingness comparison plot, and (3) ranked missing-versus-present target-rate table/plot. The output will recommend “retain as value,” “impute,” “impute plus indicator,” or “exclude/review,” but will not claim a causal missingness mechanism.
- **Dependency created:** Supplies the availability component of Q2c and the missing-data treatment needed for later comparisons.

### Q2b. How much of the 122 columns is actually redundant?

- **Status:** Partially answerable. Statistical redundancy in this sample is measurable; semantic equivalence and redundancy after future feature engineering are not fully determined by pairwise association.
- **Method:** Remove IDs and `TARGET`; use the cleaned candidate representation from S4–S6. Identify exact duplicate/constant columns and group numeric columns connected at absolute Spearman correlation `|ρ| ≥ 0.90`. Within each group, retain the member with the highest non-missing coverage; break coverage ties in favor of the more interpretable field. Record every group member rather than silently discarding it. Assess categorical association separately with an appropriate measure such as Cramér’s V, but do not apply the numeric `ρ` threshold to categorical fields. Review known feature families and shared suffixes (`_AVG`, `_MODE`, `_MEDI`) together.
- **Tables needed:** `application_train`; `application_test` only to check that a proposed representative is usable in both splits.
- **Output:** A compact association heatmap, a redundancy-group table listing group members and a proposed representative, and a before/after count of usable feature groups. Large 122-by-122 tables should not be printed in full.
- **Dependency created:** Prevents duplicate measurements from dominating the univariate ranking and final screen.

### Q1 / Q1a. Which individual applicant characteristics separate borrowers with repayment difficulty from those without it, and is any one strong enough to stand alone?

- **Status:** Partially answerable. The approved stand-alone criterion makes the sample testable, but no single-variable association proves causality, legal usability, or production performance.
- **Description flag:** **“1a. Single-variable separation” is a section label, not a question.** It is operationalized by the question above. The Q1 parent is an umbrella question; this item provides its main test.
- **Method:** Use only leakage-safe, non-ID candidates from S5 and the treatments from S4–S6; exclude fields with more than 60% missing from this ranking. Use a fixed held-out fold established before comparing variables. For numeric fields, compare distributions by target, target rates across deciles, univariate ROC AUC with direction normalized, and worst-decile-to-best-decile default-rate ratio. For categorical fields, report supported category rates and a held-out univariate encoding/score so high-cardinality variables are not rewarded for overfitting. Any reported bin or category must contain at least 1,000 applicants; smaller categorical levels are pooled into `Other`, and binning must be reduced if needed to meet the same minimum. Rank effect size alongside coverage and uncertainty, and annotate redundant groups from Q2b. A variable stands alone only if held-out univariate AUC is at least 0.70 **and** its worst-decile default rate is at least 2.0 times its best-decile rate.
- **Pre-registered expectation:** No single variable will achieve held-out AUC ≥ 0.70; the `EXT_SOURCE_*` fields are expected to come closest. Record whether this expectation is supported without changing the thresholds after seeing results.
- **Tables needed:** `application_train`; `application_test` only for availability and range/level checks, never for target performance.
- **Output:** A ranked univariate-results table with variable, type, nonmissing coverage, held-out AUC, worst/best-decile default-rate ratio, stand-alone pass/fail, uncertainty, and redundancy group; followed by small-multiple plots for only the strongest supported variables.
- **Dependency created:** Supplies the predictive-value component of Q2c.

### Q1b / Q4. Do the money variables behave the way an underwriter would expect?

- **Status:** Partially answerable. Observed direction and monotonicity are testable, but underwriting expectations must be stated in advance. Complete affordability cannot be measured because verified expenses, full obligations, the current loan term, and APR are absent.
- **Duplicate flag:** Q1b and Q4 are the same question. Analyze them once here and cross-reference the single result; do not create duplicate outputs.
- **Method:** Pre-register directional expectations, such as lower difficulty with higher income and higher difficulty with larger payment/income or credit/income burdens. Examine `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, and `AMT_GOODS_PRICE`, plus carefully defined ratios such as annuity-to-income, credit-to-income, and credit-to-goods-price. Handle zero/missing denominators and S4 anomalies explicitly. Preserve raw income values in every numerical summary and table; cap income at the training 99.9th percentile for plotting only and label the plot axis accordingly. Plot target rate by supported quantile bins of at least 1,000 applicants, report bin counts and uncertainty, and flag non-monotonic or sparse tails. Treat results as associations, not underwriting rules.
- **Tables needed:** `application_train`; corresponding columns in `application_test` for distribution-shift checks.
- **Output:** A definition table for each money measure/ratio and its expected direction; binned count/default-rate tables with confidence intervals; line or point plots of rate by bin; and a concise expectation-versus-observation summary.
- **Dependency created:** Adds business-coherence evidence to the three-way screen in Q2c.

### Q2 / Q2c. Which predictors survive all three screens: predictive value, availability, and verifiability?

- **Status:** Partially answerable. Predictive value and observed availability are measurable. **Verifiability cannot be established from these CSVs** because they contain no verification status, source documents, or collection costs. `feature_report.md` can support only a documented qualitative rating (direct application field, opaque external score, proxy, or unavailable), not proof of operational verifiability.
- **Description flag:** **“2c. The three-way screen” is a section label, not a question.** The Q2 parent supplies the question; Q2a and Q2b are prerequisite analyses.
- **Method:** Combine Q1 predictive evidence, S6 availability, Q2b redundancy groups, Q1b business coherence, train/test usability from S3, timing from S5, and the approved qualitative rubric grounded in `feature_report.md`. Assign **Tier A** to document-backed fields (including income, employment, age, education, family status, housing type, contract amounts, and car ownership); **Tier B** to self-reported or structural fields (including contact flags, phone availability, region ratings, and process timing); and **Tier C** to opaque fields (`EXT_SOURCE_*` and social-circle counts). Assign **Pass** only to Tier A fields with at least 80% coverage and meaningful Q1 signal; **Review** to Tier B fields or Tier A fields with 50–80% coverage; and **Fail** to Tier C fields or any field below 50% coverage. Report the predictive strength of Fail variables, including the expectedly strong opaque fields, so the cost of exclusion remains visible. Keep predictive performance distinct from fairness or legal permissibility; sensitive/proxy fields should be flagged for governance review even if predictive.
- **Tables needed:** Results produced by S3–S6, Q2b, Q1, and Q1b; `application_train`, `application_test`; `feature_report.md` for qualitative provenance/verifiability limitations.
- **Output:** A feature scorecard with one row per feature group and columns for validation strength, coverage, train/test compatibility, anomaly burden, temporal status, redundancy, verifiability tier (A/B/C), governance flag, and final Pass/Review/Fail disposition; plus a short survivor list with reasons. Fail-tier predictive strength remains visible in the same table.
- **Dependency created:** Defines the focused candidate set that a later modeling stage may use.

### Q3. Does the repayment-difficulty rate vary enough across major customer segments to threaten the stability promised by the BPS?

- **Status:** Partially answerable. This EDA can identify cross-sectional differences in this training sample using the approved materiality rules. It **cannot answer temporal or future-population stability**, because there is no labeled time-based or external validation sample, and test has no `TARGET`.
- **Method:** Predefine major segments and avoid data-driven cherry-picking: `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE`, `REGION_RATING_CLIENT_W_CITY`, and `NAME_HOUSING_TYPE`, subject to S3–S6 data-quality results. Pool levels below 1,000 applicants into `Other`. For every supported level, report applicant count/share, target count/rate, 95% confidence interval, and absolute and relative difference from the overall rate. **Note** a segment only when `n ≥ 1,000`, its confidence interval excludes the overall rate, and `segment rate / overall rate ≥ 1.25` or `≤ 0.80`. Mark a result as **threatening stability** when those support and confidence conditions hold and the ratio is `≥ 2.0` or `≤ 0.50`, or when a predictor's direction reverses between sufficiently large segments. Treat protected/proxy attributes as audit dimensions, not automatic underwriting inputs.
- **Tables needed:** `application_train`; `application_test` only for segment-prevalence comparison.
- **Output:** Segment profile table; dot/interval plots of target rate by segment; train/test segment-share comparison; and, if sample sizes allow, a predictor-by-segment stability matrix showing direction/effect consistency and uncertainty.
- **Decision rule:** The 1.25× note threshold, 2.0× threat threshold, `n ≥ 1,000` requirement, confidence-interval rule, and direction-reversal rule are pre-registered and must not be changed after viewing results.

## Items the current wording cannot fully answer

No submitted question is wholly impossible with the in-scope data, but several claims must be narrowed:

- Q1 can apply the approved stand-alone criterion in a held-out fold, but it still cannot establish production performance from EDA alone.
- Q1b/Q4 cannot test complete affordability or reconstruct current-loan pricing because current term, APR, verified expenses, and complete obligations are absent.
- Q2 cannot prove operational verifiability; it can only apply a transparent qualitative rubric from the documentation.
- Q3 cannot establish stability over time or in future populations because no labeled temporal/external sample is available.
- S5 cannot prove the provenance or exact availability timestamp of opaque external scores.

These limits should remain visible in the notebook so descriptive associations are not presented as causal, operational, or policy conclusions.

## Descriptions and duplicated items in the submitted list

- **“The Standing Checks”** and its bullets are requirements/descriptions, not analytic questions.
- **“1a. Single-variable separation”** is a description/section label; it has been rewritten as an operational question under Q1.
- **“2c. The three-way screen”** is a description/section label; it has been tied to the Q2 parent question.
- Q1 and Q2 are umbrella questions. Their subquestions provide the executable analyses and should not be reported as unrelated findings.
- Q1b and Q4 are exact duplicates and should produce one analysis, not two.

## Approved decision rules

The anomaly treatments, redundancy and reporting thresholds, stand-alone definition, verifiability rubric, and segment-stability rules above are approved and pre-registered in `docs/approvals.md`. They should be implemented exactly as written and reported even when results conflict with the pre-registered expectation. All transformations must occur in derived DataFrames; source CSVs must remain unchanged.

## D1. Approval-threshold trade-off — deferred to modeling

D1 is **deferred, not omitted**. It asks what the strongest predictors imply for an approval threshold and the resulting approval-rate/repayment-difficulty-rate trade-off. Answering it requires a composite model score evaluated out of sample; fitting and assessing that composite on the same observations during EDA would yield an optimistically biased trade-off curve.

The modeling stage must answer D1 using out-of-sample scores, a held-out validation split, and a stated cost ratio between a missed default and a declined good applicant. This EDA may identify candidate predictors and describe univariate separation, but it must not present a composite approval threshold as an exploratory result.
