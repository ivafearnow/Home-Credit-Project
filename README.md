# home-credit-project

Practice Capstone

Ivania Fearnow

## Data Preparation

`src/features/data_preparation.py` creates reproducible, application-level training and test datasets for Home Credit default modeling. It preserves raw CSVs, fits parameters on training data only, applies them to test data, safely aggregates bureau history, saves prepared files, and validates the final schema.

### What the script does

| Transformation | EDA decision # | Reason |
| --- | ---: | --- |
| Use copied DataFrames; do not alter raw CSVs. | 1 | Preserves auditability and reproducibility. |
| Retain one row per application and check `SK_ID_CURR` uniqueness. | 2 | The dataset has application grain and no person identifier for customer deduplication. |
| Keep `TARGET` only as the final training outcome, outside the predictor schema. | 3 | The outcome is unavailable for a new applicant. |
| Retain `SK_ID_CURR` for tracking; downstream models must exclude it. | 4 | Identifiers can cause memorization rather than generalization. |
| Convert `DAYS_EMPLOYED = 365243` to missing and retain the row. | 5 | The value is a placeholder, not an employment duration. |
| Create `FLAG_NOT_EMPLOYED` before replacing the sentinel. | 6 | The pensioner-heavy sentinel may contain useful signal. |
| Drop training `CODE_GENDER = "XNA"` rows but retain test rows. | 7 | There are four undocumented training rows, while every test applicant must be scored. |
| Replace `REGION_RATING_CLIENT_W_CITY = -1` with the training mode. | 8 | The valid domain is `1`, `2`, or `3`; repair preserves the test row. |
| Retain raw `AMT_INCOME_TOTAL` extremes. | 9 | The EDA cap was for plotting only. |
| Retain suspect social-circle counts without headline claims. | 10 | No documented repair rule exists. |
| Add selected missingness indicators. | 11 | Missingness, especially for building fields, may be predictive. |
| Do not apply the general missingness treatment thresholds. | 12 | They are recommendations, not a fitted production rule. |
| Do not automatically drop high-missingness building fields. | 13 | This was a modeling availability screen. |
| Do not automatically drop constant `FLAG_DOCUMENT_*` fields. | 14 | This is a later model-selection decision. |
| Leave train-only category handling to the encoder. | 15 | The EDA requires explicit handling at encoding time. |
| Do not pool categories into `Other` or create `<MISSING>` categories. | 16 | This rule was used only for analysis and segmentation. |
| Do not use target-aware median imputation or target means. | 17 | These were held-out univariate-screen calculations. |
| Do not exclude features below 40% coverage. | 18 | The threshold was used only in univariate scoring. |
| Do not collapse correlated numeric features. | 19 | Correlation grouping belongs to the feature scorecard. |
| Do not remove constant fields. | 20 | Constant handling belongs to later feature screening. |
| Create affordability and loan-structure ratios. | 21 | Payment burden and financing structure can distinguish repayment capacity. |
| Use missing values for ratios with non-positive denominators. | 22 | Such ratios are financially meaningless. |
| Bin `AMT_CREDIT` with training-only quantile boundaries. | 23 | Loan-size default risk is non-monotonic. |
| Retain raw `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` values. | 24 | Their exclusion is a governance-based feature-selection decision. |
| Retain application-time and bureau-enquiry fields. | 25 | The EDA considers them potentially available at application time. |
| Do not assign Pass, Review, or Fail labels. | 26 | Labels are feature-scorecard output. |
| Keep `DAYS_BIRTH` and derive `AGE_YEARS`. | 27 | Age is a candidate subject to governance review; years are interpretable. |
| Do not resolve protected-attribute or proxy concerns. | 28 | These require legal and compliance review. |
| Aggregate `bureau.csv` to one row per `SK_ID_CURR`, then left-join it. | 29 | An application-date-safe aggregate prevents row multiplication and future leakage. |
| Do not repair or delete other audited anomalies. | 30 | The EDA requires review or flagging, not unsupported automatic changes. |

#### EDA decisions not implemented in this script

The following decisions are deliberately excluded because they are plotting,
analysis, encoding, feature-selection, or governance work rather than repeatable
row-level preparation:

- Decision `9`: plot clipping is a visualization-only rule.
- Decision `10`: excluding suspect social-circle values from headline claims is a reporting rule.
- Decision `12`: the missingness thresholds are recommendation logic, not a fitted production rule.
- Decision `13`: the building-field availability screen is a modeling selection rule.
- Decision `14`: removing constant `FLAG_DOCUMENT_*` fields belongs to model selection.
- Decision `15`: train-only category handling belongs at encoding time.
- Decision `16`: pooling categories and representing missingness as `<MISSING>` was used only for analysis and segmentation.
- Decision `17`: target-aware median imputation and target means were held-out univariate-screen calculations.
- Decision `18`: the 40% coverage threshold was used only in univariate scoring.
- Decision `19`: correlation grouping belongs to the feature scorecard.
- Decision `20`: constant-field removal belongs to feature screening.
- Decision `24`: excluding opaque external scores is a governance-based feature-selection decision.
- Decision `26`: Pass, Review, and Fail labels are feature-scorecard output.
- Decision `28`: protected-attribute and proxy issues require legal and compliance review.

Decisions `3`, `4`, `25`, `27`, and `30` are recorded as retention constraints
rather than destructive transformations.

### Features created

- **Age/employment years:** `AGE_YEARS` and `EMPLOYMENT_YEARS` express life stage and job tenure. Both may proxy income security and repayment resilience.
- **Financial ratios:** `ANNUITY_TO_INCOME`, `CREDIT_TO_INCOME`, and `CREDIT_TO_GOODS_PRICE` measure payment burden, requested debt burden, and financing structure. These may separate applicants with different capacity to repay.
- **Missing indicators:** `DAYS_EMPLOYED_MISSING`, `AMT_ANNUITY_MISSING`, `AMT_GOODS_PRICE_MISSING`, `OWN_CAR_AGE_MISSING`, `OCCUPATION_TYPE_MISSING`, `HOUSETYPE_MODE_MISSING`, `EXT_SOURCE_1_MISSING`, `EXT_SOURCE_2_MISSING`, and `EXT_SOURCE_3_MISSING` preserve signal in absent values.
- **Interaction terms:** `ANNUITY_TO_INCOME_X_EMPLOYMENT_YEARS` combines payment burden and tenure because equal payment burdens can imply different resilience for applicants with different job stability.
- **Binned variables:** `AMT_CREDIT_BIN` represents the observed humped relationship between loan size and default risk.
- **Bureau aggregates:** `BUREAU_RECORD_COUNT`, `BUREAU_UNIQUE_CREDIT_COUNT`, `BUREAU_ACTIVE_CREDIT_COUNT`, and `BUREAU_CLOSED_CREDIT_COUNT`; plus `MEAN`, `MAX`, and `SUM` for `DAYS_CREDIT`, `CREDIT_DAY_OVERDUE`, `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `AMT_CREDIT_MAX_OVERDUE`, `CNT_CREDIT_PROLONG`, `AMT_CREDIT_SUM`, `AMT_CREDIT_SUM_DEBT`, `AMT_CREDIT_SUM_LIMIT`, `AMT_CREDIT_SUM_OVERDUE`, `DAYS_CREDIT_UPDATE`, and `AMT_ANNUITY`. All are prefixed with `BUREAU_` and summarize prior-credit volume, status, debt, overdue behavior, and exposure that may predict default.
- **Cleaning flag:** `FLAG_NOT_EMPLOYED` records the pensioner-heavy employment sentinel before it is replaced.

### Train/test consistency

The script learns these parameters from cleaned training data only: `region_rating_client_w_city_mode`, `credit_amount_bin_edges`, `missing_indicator_columns`, and the ordered `feature_columns` schema. It stores them as `CleaningParameters`, `FeatureEngineeringParameters`, and `PreparationArtifacts` in `models/data_preparation_parameters.pkl`, then reuses them unchanged on test data. This prevents test data from influencing the region repair, bin boundaries, indicator schema, or column order.

Latest run checks:

```text
Columns match apart from TARGET: True
SK_ID_CURR is unique in prepared train: True
SK_ID_CURR is unique in prepared test:  True
```

### How to run

Run the following command from the repository root, `Home-Credit-Project`:

```bash
python src/features/data_preparation.py
```

For use from a notebook, call the functions in this order:

```python
import pandas as pd

from src.features.data_preparation import (
    clean_application_data,
    engineer_application_features,
    fit_cleaning_parameters,
    fit_feature_engineering_parameters,
    prepare_data,
)

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")
bureau = pd.read_csv("data/raw/bureau.csv")

# One-call pipeline: aggregate bureau, fit on train, and apply to test.
prepared_train, prepared_test, artifacts = prepare_data(
    train, test, bureau_df=bureau
)

# Equivalent application-only order for inspecting individual stages.
cleaning_parameters = fit_cleaning_parameters(train)
cleaned_train = clean_application_data(train, cleaning_parameters, split="train")
cleaned_test = clean_application_data(test, cleaning_parameters, split="test")
feature_parameters = fit_feature_engineering_parameters(cleaned_train)
engineered_train = engineer_application_features(cleaned_train, feature_parameters)
engineered_test = engineer_application_features(cleaned_test, feature_parameters)
```

### Inputs and outputs

| Type | File | Purpose or dimensions from the latest run |
| --- | --- | --- |
| Input | `data/raw/application_train.csv` | Raw training applications; includes `TARGET`. |
| Input | `data/raw/application_test.csv` | Raw scoring applications; excludes `TARGET`. |
| Input | `data/raw/bureau.csv` | Prior-credit records aggregated to applicant level. |
| Output | `data/processed/train_prepared.csv` | `307,507 × 179` rows × columns; includes `TARGET`. |
| Output | `data/processed/test_prepared.csv` | `48,744 × 178` rows × columns; matches training predictors except `TARGET`. |
| Output | `models/data_preparation_parameters.pkl` | Serialized training-fitted parameters and final feature-column order. |

All data and model files are gitignored, including the raw inputs, prepared CSVs, and serialized artifact.

### Script

[src/features/data_preparation.py](src/features/data_preparation.py)
