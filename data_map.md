# Home Credit data map

## Scope and conventions

This is a structural data audit, not exploratory data analysis. It covers the nine data/output CSVs in `data/raw/`; `docs/HomeCredit_columns_description.csv` is treated as metadata rather than as a business table. The same dictionary also appears in `data/raw/`, and the two copies are byte-for-byte identical.

- Row counts exclude the header.
- Keys are inferred from the dictionary and observed grain because the CSVs do not declare database constraints.
- Cardinality is written from child to parent. `M:1` means many child rows may join to one parent row; the reverse relationship is `1:0..M` unless stated otherwise.
- “Train coverage” is the number of distinct `application_train.SK_ID_CURR` values represented by at least one row, divided by 307,511 train applicants. For `bureau_balance`, coverage is measured indirectly through `bureau`.
- A “value contradiction” is limited to an observed value that conflicts with an explicit domain or time meaning in the dictionary. Undocumented categories and schema discrepancies are also called out, but are not silently treated as domain violations.

## Summary

| Table | Grain | Rows | Inferred primary key | Unique? | Train applicants represented |
|---|---|---:|---|---|---:|
| `application_train` | One current Home Credit application/applicant | 307,511 | `SK_ID_CURR` | Yes | 307,511 (100.00%) |
| `application_test` | One held-out current Home Credit application/applicant | 48,744 | `SK_ID_CURR` | Yes | 0 (0.00%) |
| `bureau` | One credit reported by the Credit Bureau for a current applicant | 1,716,428 | `SK_ID_BUREAU` | Yes | 263,491 (85.69%) |
| `bureau_balance` | One monthly status snapshot of one bureau credit | 27,299,925 | (`SK_ID_BUREAU`, `MONTHS_BALANCE`) | Yes | 92,231 (29.99%), indirectly |
| `previous_application` | One previous Home Credit application | 1,670,214 | `SK_ID_PREV` | Yes | 291,057 (94.65%) |
| `POS_CASH_balance` | One monthly POS/cash-loan snapshot for a previous credit | 10,001,358 | (`SK_ID_PREV`, `MONTHS_BALANCE`) | Yes | 289,444 (94.12%) |
| `credit_card_balance` | One monthly credit-card snapshot for a previous credit | 3,840,312 | (`SK_ID_PREV`, `MONTHS_BALANCE`) | Yes | 86,905 (28.26%) |
| `installments_payments` | One observed payment record against a scheduled installment | 13,605,401 | No unique row key supplied; see below | No | 291,643 (94.84%) |
| `sample_submission` | One prediction placeholder for one test application | 48,744 | `SK_ID_CURR` | Yes | 0 (0.00%) |

## Table details

### `application_train.csv`

- **Grain and rows:** one current Home Credit application/applicant; 307,511 rows.
- **Primary key:** `SK_ID_CURR`; all 307,511 values are unique.
- **Foreign keys:** none.
- **Train coverage:** 307,511 / 307,511 = **100.00%**.
- **Value contradictions:** `DAYS_EMPLOYED` is defined as how many days *before* the application employment began, but 55,374 rows contain the positive value `365243`. This is an undocumented sentinel, not a plausible relative-day value. `CODE_GENDER` also contains the undocumented category `XNA` in 4 rows; because the dictionary gives no allowed category list, this is an undocumented value rather than a strict domain violation.

### `application_test.csv`

- **Grain and rows:** one held-out current Home Credit application/applicant; 48,744 rows.
- **Primary key:** `SK_ID_CURR`; all 48,744 values are unique. The IDs are disjoint from train.
- **Foreign keys:** none.
- **Train coverage:** 0 / 307,511 = **0.00%**.
- **Value contradictions:** `DAYS_EMPLOYED` contains the same positive `365243` sentinel in 9,274 rows. `REGION_RATING_CLIENT_W_CITY`, whose documented domain is 1, 2, or 3, equals `-1` in 1 row.

### `bureau.csv`

- **Grain and rows:** one Credit Bureau credit associated with a current applicant; 1,716,428 rows.
- **Primary key:** `SK_ID_BUREAU`; all 1,716,428 values are unique.
- **Foreign keys:** `SK_ID_CURR` -> the union of `application_train.SK_ID_CURR` and `application_test.SK_ID_CURR`, **M:1**. All rows resolve; each application has zero or many bureau credits.
- **Train coverage:** 263,491 / 307,511 = **85.69%**.
- **Value contradictions:** `DAYS_CREDIT_UPDATE` is documented as the number of days *before* the current application when the bureau information arrived, but 17 rows are positive (maximum `372`).

### `bureau_balance.csv`

- **Grain and rows:** one monthly balance/status snapshot for one bureau credit; 27,299,925 rows.
- **Primary key:** (`SK_ID_BUREAU`, `MONTHS_BALANCE`); all 27,299,925 combinations are unique.
- **Foreign keys:** `SK_ID_BUREAU` -> `bureau.SK_ID_BUREAU`, expected **M:1**. The relationship is only partial in the supplied extracts: 24,179,741 rows resolve and 3,120,184 rows do not. A bureau credit has zero or many monthly snapshots among the matching records.
- **Train coverage:** 92,231 / 307,511 = **29.99%**, found by joining through the available `bureau` rows. Because some bureau IDs have no supplied parent, this is the measurable coverage, not necessarily the coverage of the source system.
- **Value contradictions:** `MONTHS_BALANCE` equals `0` in 610,965 rows even though this table's dictionary entry says `-1` is the freshest balance date. Observed values range from `-96` through `0`.

### `previous_application.csv`

- **Grain and rows:** one previous Home Credit application, whether or not it became a credit; 1,670,214 rows.
- **Primary key:** `SK_ID_PREV`; all 1,670,214 values are unique.
- **Foreign keys:** `SK_ID_CURR` -> the union of `application_train.SK_ID_CURR` and `application_test.SK_ID_CURR`, **M:1**. All rows resolve; each current application has zero or many previous applications.
- **Train coverage:** 291,057 / 307,511 = **94.65%**.
- **Value contradictions:** the undocumented sentinel `365243` appears in fields documented as relative dates: `DAYS_FIRST_DRAWING` (934,444 rows), `DAYS_FIRST_DUE` (40,645), `DAYS_LAST_DUE_1ST_VERSION` (93,864), `DAYS_LAST_DUE` (211,221), and `DAYS_TERMINATION` (225,913). The value is about 1,000 years and therefore is not a literal relative-day measurement.

### `POS_CASH_balance.csv`

- **Grain and rows:** one monthly POS/cash-loan balance snapshot for one previous credit; 10,001,358 rows.
- **Primary key:** (`SK_ID_PREV`, `MONTHS_BALANCE`); all 10,001,358 combinations are unique.
- **Foreign keys:** `SK_ID_CURR` -> the union of train and test applications, **M:1**, with all rows resolved. `SK_ID_PREV` -> `previous_application.SK_ID_PREV`, expected **M:1**, is partial: 9,660,797 rows resolve and 340,561 do not. For every row whose `SK_ID_PREV` resolves, its `SK_ID_CURR` agrees with the parent application.
- **Train coverage:** 289,444 / 307,511 = **94.12%**.
- **Value contradictions:** none found against explicit dictionary domains or meanings.

### `credit_card_balance.csv`

- **Grain and rows:** one monthly credit-card balance snapshot for one previous credit; 3,840,312 rows.
- **Primary key:** (`SK_ID_PREV`, `MONTHS_BALANCE`); all 3,840,312 combinations are unique.
- **Foreign keys:** `SK_ID_CURR` -> the union of train and test applications, **M:1**, with all rows resolved. `SK_ID_PREV` -> `previous_application.SK_ID_PREV`, expected **M:1**, is partial: 2,757,496 rows resolve and 1,082,816 do not. All resolved previous-credit rows agree on `SK_ID_CURR`.
- **Train coverage:** 86,905 / 307,511 = **28.26%**.
- **Value contradictions:** none found against explicit dictionary domains or meanings.

### `installments_payments.csv`

- **Grain and rows:** one observed payment record against a scheduled installment of a previous credit; 13,605,401 rows. Multiple records can apply to the same scheduled installment, consistent with split or repeated payments.
- **Primary key:** no unique row-level key is supplied. The natural schedule key (`SK_ID_PREV`, `NUM_INSTALMENT_VERSION`, `NUM_INSTALMENT_NUMBER`) has 12,951,918 distinct combinations and 653,483 excess duplicate occurrences, so it is **not unique** and cannot serve as this table's primary key.
- **Foreign keys:** `SK_ID_CURR` -> the union of train and test applications, **M:1**, with all rows resolved. `SK_ID_PREV` -> `previous_application.SK_ID_PREV`, expected **M:1**, is partial: 12,354,575 rows resolve and 1,250,826 do not. All resolved previous-credit rows agree on `SK_ID_CURR`.
- **Train coverage:** 291,643 / 307,511 = **94.84%**.
- **Value contradictions:** none found against explicit dictionary domains or meanings.

### `sample_submission.csv`

- **Grain and rows:** one prediction placeholder per test application; 48,744 rows.
- **Primary key:** `SK_ID_CURR`; all values are unique.
- **Foreign keys:** `SK_ID_CURR` -> `application_test.SK_ID_CURR`, **1:1**. Its ID set exactly equals the test ID set.
- **Train coverage:** 0 / 307,511 = **0.00%**.
- **Value contradictions:** not assessable from the supplied dictionary because `sample_submission.csv` is not documented there.

## Dictionary/schema discrepancies

- The dictionary uses `SK_BUREAU_ID` in `bureau` and `bureau_balance`; both actual CSVs use `SK_ID_BUREAU`.
- The dictionary lists `NFLAG_MICRO_CASH` for `previous_application`, but that column is absent from the CSV.
- The shared `application_{train|test}` dictionary lists `TARGET`; its absence from `application_test` is expected for a held-out scoring table, not treated as an error.
- `sample_submission.csv` has no entry in the dictionary.
