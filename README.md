# home-credit-project

Practice Capstone

Ivania Fearnow

## Data Preparation

`src/features/data_preparation.py` prepares the Home Credit application data
for modeling without changing the source CSV files. It fits all learned rules
on the training data, applies those same rules to test data, and verifies that
the resulting predictor columns match.

### Pipeline steps

1. Fit the valid `REGION_RATING_CLIENT_W_CITY` replacement mode from training
   data only.
2. Clean both application tables: flag and replace the `DAYS_EMPLOYED = 365243`
   placeholder, remove training-only `CODE_GENDER = XNA` records, and repair the
   invalid region rating.
3. Create financial ratios, positive age and employment-tenure features,
   missingness indicators, an employment/payment-burden interaction, and a
   binned credit-amount feature.
4. Learn credit-amount bin boundaries and the missingness-indicator schema from
   training data only, then reuse them for test data.
5. Optionally aggregate `bureau.csv` to one row per `SK_ID_CURR` and left-join
   those features to the application tables.
6. Confirm that train and test have the same predictor columns and order;
   `TARGET` remains only in training data.
7. Save the fitted cleaning values, feature settings, and final feature order.

### Required data layout

The runnable example expects these files under `data/raw/`:

```text
data/raw/application_train.csv
data/raw/application_test.csv
```

`application_train.csv` must include `TARGET`; `application_test.csv` must not.
To use bureau-history features, also provide `data/raw/bureau.csv` and pass its
loaded DataFrame as `bureau_df` to `prepare_data()`.

### Run the included example

From the `Home-Credit-Project` directory:

```bash
python src/features/data_preparation.py
```

The script prints the prepared train/test dimensions, whether the predictor
columns match apart from `TARGET`, and the number of remaining missing values.

### Use from Python

```python
import pandas as pd

from src.features.data_preparation import prepare_data

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")

prepared_train, prepared_test, artifacts = prepare_data(train, test)
```

To include bureau features:

```python
bureau = pd.read_csv("data/raw/bureau.csv")
prepared_train, prepared_test, artifacts = prepare_data(
    train,
    test,
    bureau_df=bureau,
)
```

### Outputs

- `prepared_train`: prepared modeling data with `TARGET` as its final column.
- `prepared_test`: prepared scoring data with the same predictor columns and
  order as `prepared_train` excluding `TARGET`.
- `artifacts`: the fitted cleaning parameters, feature-engineering parameters,
  and final predictor schema returned by `prepare_data()`.
- `models/data_preparation_parameters.pkl`: the saved fitted artifacts created
  by the default run.
