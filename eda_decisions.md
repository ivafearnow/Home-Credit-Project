# EDA decision inventory — `01_eda.ipynb`

Source: `Home-Credit-Project/notebooks/01_eda.ipynb`. “Applied” means the notebook changes a derived DataFrame; source CSVs remain unchanged. Cell numbers below are zero-based notebook cell indices.

## Data-cleaning and transformation decisions

1. **Keep the raw CSV files unchanged; make all repairs in copied/derived frames.** **Applied.** The notebook says, “All repairs are applied to derived DataFrames; the source CSVs are not modified.” (cell 1; also cell 2).

2. **Treat each row as an application, not a customer, and retain the full application-level population.** The key is unique and no full-row duplicates are found, but repeat applicants cannot be identified because no person identifier exists. (cells 4, 8–9). This is a grain/interpretation decision rather than a row-removal rule.

3. **Exclude `TARGET` from predictor candidates.** **Applied in analysis.** Candidate/model columns are defined from test columns, and `SK_ID_CURR` is excluded: `model_cols = [c for c in test.columns if c != "SK_ID_CURR"]`; because test lacks `TARGET`, it is not a candidate feature. The timing inventory separately marks `TARGET` “exclude.” (cells 26 and 20).

4. **Exclude `SK_ID_CURR` from predictors.** **Applied in analysis.** It is treated as an identifier with “identifier memorization” risk and marked “exclude.” (cell 20; implementation in cell 26).

5. **Recode the `DAYS_EMPLOYED = 365243` sentinel to missing and retain the rows.** **Applied to train and test derived frames.** The anomaly register specifies “set NaN; add `FLAG_NOT_EMPLOYED`; retain rows,” and the code replaces the sentinel with `NaN`. (cell 17). The rationale is that it affects about 18% of rows and is largely a pensioner marker, so dropping it would change the study population. (cell 18).

6. **Create `FLAG_NOT_EMPLOYED` from the `DAYS_EMPLOYED` sentinel.** **Applied to train and test derived frames** as an `int8` flag before the sentinel is replaced. The notebook cautions that this flag is better interpreted as a pensioner marker than a general unemployment indicator. (cells 17–18; timing decision to keep it in cell 20).

7. **Drop training rows where `CODE_GENDER = "XNA"`; do not drop test rows.** **Applied.** The anomaly register says “drop train rows; never drop test rows,” and the code filters the train frame accordingly. The stated reason is that there are only four training rows, whereas every test applicant must be scored. (cells 17–18).

8. **Repair the invalid test value `REGION_RATING_CLIENT_W_CITY = -1` with the training mode.** **Applied to the derived test frame.** The dictionary domain is 1, 2, or 3; the register says “replace test value with training mode,” and the code does so. (cell 17). The notebook explicitly chooses repair rather than deletion because every test applicant needs a prediction. (cell 18).

9. **Retain raw `AMT_INCOME_TOTAL` extremes; cap only their visualization at the training 99.9th percentile.** **Applied for plots only.** The anomaly register says “retain raw; cap plots only at training p99.9,” and the income histogram uses `clip(upper=income_plot_cap)`. (cells 17 and 32).

10. **Retain unusually large social-circle counts, flag them as suspect, and exclude them from headline claims.** The register’s treatment for `OBS_*`/`DEF_*_CNT_SOCIAL_CIRCLE` is “retain; flag; exclude from headline claims.” (cell 17). The univariate output implements headline ineligibility for those columns. (cell 29).

11. **Treat missing building-characteristic values as signal: use missingness indicators rather than average/mean imputation.** **Proposed for modeling; not implemented as new indicator columns in this notebook.** The notebook says, “If I filled these in with an average I would erase whatever signal the absence carries, so I would add a flag marking the value as missing instead.” (cell 24). The final results repeat “missing-indicators rather than mean imputation.” (cell 44).

12. **Use a missingness treatment rule for other fields: impute plus an indicator when missingness has a sufficiently large target-rate association; impute when it does not; exclude/review tiny or very-high-missingness groups.** **Recommendation logic, not a fitted preprocessing pipeline.** The rule is: tiny group `< 1,000` → “exclude/review”; train missingness `>60%` → “exclude/review”; absolute missing-versus-present target-rate difference `>=0.005` → “impute plus indicator”; otherwise nonzero missingness → “impute.” (cell 23).

13. **Treat building fields with roughly 70% missingness as failing the availability screen.** **Screening decision.** The notebook says these fields “fail the availability test” because they are empty for roughly 70% of applicants. (cell 24).

14. **Remove/avoid eleven `FLAG_DOCUMENT_*` columns that are constant in the scoring population.** **Decision for later scoring/model selection, not an explicit `drop` statement in the EDA code.** “They cannot separate anyone in the population I actually have to score.” (cell 15; summarized in cell 44).

15. **Handle the three train-only categorical levels explicitly at encoding time.** **Proposed.** The levels are `CODE_GENDER = XNA`, unknown family status, and maternity leave. The notebook notes that an encoder fitted on training would yield an all-zero column in test, and the final audit says they “need explicit handling at encode time.” (cells 15 and 44).

16. **For categorical univariate/segment analysis, pool levels with fewer than 1,000 observations into `Other`; represent missing as `<MISSING>`.** **Applied to those analyses.** `supported_groups` makes this recode, and the notebook uses it for categorical scoring and segmentation. (cells 29 and 38).

17. **For numeric univariate and money-variable analyses, fill missing values using the fit-split median; use fit-split category target means with global-mean fallback for categorical scores.** **Applied only to the held-out univariate-screen calculations.** Numeric scores use `fit_s.median()`; categorical scores pool unsupported/unknown levels to `Other` and fall back to the fit-set global target mean. (cell 29). This is evaluation-specific imputation/encoding, not a saved production transformation.

18. **Exclude predictor candidates with under 40% nonmissing coverage from univariate scoring.** **Applied to the univariate screen.** The code continues without scoring a column when `coverage < .40`. (cell 29).

19. **Collapse highly redundant numeric features into correlation groups and retain one representative per group.** **Applied to the feature scorecard.** Numeric pairs with absolute Spearman correlation `>= .90` are grouped by connected components. The representative has highest nonmissing coverage; ties prefer a name not ending `_MODE` or `_MEDI`, then a shorter/alphabetically earlier name. (cell 26). The results state 71 pairs collapse into 15 groups. (cells 27 and 44).

20. **Treat constant fields as unusable and account for them before reporting usable feature groups.** **Applied in redundancy summary/screening.** Constant columns are identified with one distinct value including missingness and removed in the “usable groups after constants and numeric grouping” count. (cell 26).

21. **Derive affordability/loan-structure features: `ANNUITY_TO_INCOME`, `CREDIT_TO_INCOME`, and `CREDIT_TO_GOODS_PRICE`.** **Applied in the `money` analysis frame.** Each ratio is formed only when its denominator is positive; otherwise it is set to missing. (cell 32).

22. **Do not form money ratios with non-positive denominators.** **Audit/treatment rule.** The additional audit records this treatment for non-positive income, credit, annuity, or goods-price values. (cell 42). The actual ratio construction enforces the positive-denominator rule. (cell 32).

23. **Model money variables with bins, splines, or a method that can learn non-monotonic effects instead of a straight-line logistic term.** **Proposed for modeling.** The notebook concludes the humped loan-size/payment-burden patterns require “binning, splines, or a model that handles non-monotonic relationships.” (cell 33; also cell 44).

24. **Exclude the opaque `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` scores from the passing feature set despite their signal.** **Feature-selection/governance decision.** Their timing/provenance is marked “review” (cell 20); their verifiability tier is C, which forces “Fail” in the scorecard (cell 35). The interpretation says they failed on verifiability alone. (cell 36).

25. **Keep/review application-time and pre-application fields, subject to the sentinel repair; retain bureau-enquiry history.** **Timing-screen decision.** Relative-date fields and `AMT_REQ_CREDIT_BUREAU_*` are marked “keep,” while general application attributes are “keep/review for governance.” (cell 20).

26. **Use availability, signal, and verifiability screens to assign feature groups to Pass, Review, or Fail.** **Applied in the scorecard.** A tier-C feature or coverage below 50% fails; tier-A, coverage at least 80%, and held-out AUC at least 0.55 passes; other cases are Review. The notebook discloses the 0.55 signal cutoff was selected after seeing results. (cell 35; caveat in cell 36).

27. **Keep four screened features as Pass candidates: `DAYS_BIRTH`, `DAYS_EMPLOYED`, `ORGANIZATION_TYPE`, and `NAME_INCOME_TYPE`.** **Feature-screen outcome.** “Four variables passed all three tests.” (cell 36).

28. **Flag, rather than resolve, protected-attribute/proxy and governance issues.** `CODE_GENDER` receives sex/protected-attribute review; `DAYS_BIRTH` age review; family, geographic, opaque-score, and social-circle fields receive related reviews. (cell 35). The notebook says legal permissibility needs compliance review. (cell 36).

29. **Defer supplementary-history aggregation to a later, application-date-safe modeling stage.** **Scope decision.** The audit calls bureau, prior applications, and repayment behavior “scoped future work” requiring application-date-safe aggregation. (cell 42).

30. **Do not repair/delete the other audited values without a documented rule.** The audit prescribes review/flagging for inconsistent household counts, car-age contradictions, implausible annuity-to-credit relations, and out-of-range external scores; it excludes contradictory social-circle values from claims. (cell 42). It reports that these checks were clean except for six car owners missing car age, treated as an availability gap. (cell 43).

## Features and feature families explicitly mentioned

1. **Identifiers/outcome:** `SK_ID_CURR`, `TARGET`. (cells 4 and 20.)
2. **Employment/age/history:** `DAYS_EMPLOYED`, `FLAG_NOT_EMPLOYED`, `DAYS_BIRTH`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`. (cells 17, 20, and 36.)
3. **Demographic, household, and housing fields:** `CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`, `NAME_HOUSING_TYPE`, `FLAG_OWN_CAR`, `OWN_CAR_AGE`, `FLAG_OWN_REALTY`. (cells 4, 35, and 37–39.)
4. **Income/employment descriptors:** `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`. (cells 17, 32, and 35–36.)
5. **Loan/affordability fields and derived ratios:** `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `ANNUITY_TO_INCOME`, `CREDIT_TO_INCOME`, `CREDIT_TO_GOODS_PRICE`. (cell 32.)
6. **External scores:** `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`. (cells 20, 36, and 42.)
7. **Building characteristics:** the 47 `_AVG`, `_MODE`, and `_MEDI` property-measure fields, including `HOUSETYPE_MODE`. (cells 4 and 24.)
8. **Document/contact and location fields:** `FLAG_DOCUMENT_2`–`FLAG_DOCUMENT_21`, contact flags, `REGION_RATING_CLIENT_W_CITY`, other region/address-consistency fields, and `LIVE_CITY_NOT_WORK_CITY`, `REG_CITY_NOT_WORK_CITY`, `REG_CITY_NOT_LIVE_CITY`. (cells 4, 15, 17, and 35.)
9. **Social-circle fields:** `OBS_*_CNT_SOCIAL_CIRCLE` and `DEF_*_CNT_SOCIAL_CIRCLE`, including 30- and 60-day versions. (cells 17 and 42.)
10. **Bureau-enquiry fields:** `AMT_REQ_CREDIT_BUREAU_*`. (cell 20.)
11. **Application-process fields:** application weekday and hour. (cell 4.)

## Supplementary tables mentioned (not used in this EDA)

1. **`installments_payments`** — covers 94.84% of training applicants; identified as one of the strongest candidates for the modeling stage. (cell 4).
2. **`previous_application`** — covers 94.65%; one of the strongest candidates. The dictionary also lists `NFLAG_MICRO_CASH`, which the notebook says is absent from the file. (cell 4).
3. **`POS_CASH_balance`** — covers 94.12%; one of the strongest candidates. (cell 4).
4. **`bureau`** — covers 85.69%. (cell 4).
5. **`bureau_balance`** — covers 29.99%; an absent row must not be interpreted as good repayment behavior. (cell 4).
6. **`credit_card_balance`** — covers 28.26%; likewise, an absent row must not be read as good repayment behavior. (cell 4).
7. **`sample_submission.csv`** — noted as having no dictionary entry. (cell 4).

