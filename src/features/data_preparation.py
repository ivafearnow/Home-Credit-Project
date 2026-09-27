"""Application-level cleaning rules established in ``01_eda.ipynb``.

This module intentionally preserves the source data: every public function
returns a copy.  Learn cleaning parameters with ``fit_cleaning_parameters``
on training data, then reuse them in ``clean_application_data`` for either
split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Literal

import numpy as np
import pandas as pd


Split = Literal["train", "test"]


@dataclass(frozen=True)
class CleaningParameters:
    """Values learned from the retained training applications only."""

    region_rating_client_w_city_mode: int | float


@dataclass(frozen=True)
class FeatureEngineeringParameters:
    """Train-fitted settings used to create engineered application features."""

    credit_amount_bin_edges: tuple[float, ...]
    missing_indicator_columns: tuple[str, ...]


@dataclass(frozen=True)
class PreparationArtifacts:
    """All training-fitted settings required to reproduce preparation."""

    cleaning: CleaningParameters
    feature_engineering: FeatureEngineeringParameters
    feature_columns: tuple[str, ...]


# These fields are important either because the EDA found their missingness
# informative or because they are core underwriting inputs whose absence should
# remain visible after a later imputation step.
MISSINGNESS_INDICATOR_CANDIDATES = (
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "OWN_CAR_AGE",
    "OCCUPATION_TYPE",
    "HOUSETYPE_MODE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)


BUREAU_NUMERIC_COLUMNS = (
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT_UPDATE",
    "AMT_ANNUITY",
)


def fit_cleaning_parameters(train_df: pd.DataFrame) -> CleaningParameters:
    """Learn cleaning values from raw training data only.

    The returned parameters must be reused when cleaning validation, test, or
    newly scored applications.  No test observations contribute to them.
    """
    _require_columns(train_df, {"CODE_GENDER", "REGION_RATING_CLIENT_W_CITY"})

    fit_frame = train_df.copy()

    # EDA decision #7: CODE_GENDER = XNA is an undocumented training-only level
    # with four rows -> remove those training rows before fitting cleaning values.
    fit_frame = fit_frame.loc[fit_frame["CODE_GENDER"] != "XNA"]

    valid_region_ratings = fit_frame.loc[
        fit_frame["REGION_RATING_CLIENT_W_CITY"].isin([1, 2, 3]),
        "REGION_RATING_CLIENT_W_CITY",
    ]
    if valid_region_ratings.empty:
        raise ValueError(
            "Cannot fit cleaning parameters: training data contains no valid "
            "REGION_RATING_CLIENT_W_CITY values in {1, 2, 3}."
        )

    # EDA decision #8: REGION_RATING_CLIENT_W_CITY = -1 is outside the documented
    # domain -> learn its replacement from the valid retained training rows only.
    region_mode = valid_region_ratings.mode().iat[0]
    return CleaningParameters(region_rating_client_w_city_mode=region_mode)


def clean_application_data(
    df: pd.DataFrame,
    parameters: CleaningParameters,
    *,
    split: Split,
) -> pd.DataFrame:
    """Apply the EDA cleaning decisions to one application-level data split.

    Parameters
    ----------
    df:
        Raw application data. It is never modified in place.
    parameters:
        Values returned by :func:`fit_cleaning_parameters` using training data.
    split:
        ``"train"`` removes the undocumented ``CODE_GENDER == "XNA"`` records.
        ``"test"`` retains every row because each applicant must be scored.

    Returns
    -------
    pandas.DataFrame
        A cleaned copy of ``df``. ``TARGET`` and ``SK_ID_CURR`` are retained;
        their later feature-exclusion rules belong to feature selection, not
        application cleaning.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")

    _require_columns(
        df,
        {
            "DAYS_EMPLOYED",
            "CODE_GENDER",
            "REGION_RATING_CLIENT_W_CITY",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
        },
    )
    cleaned = df.copy()

    # EDA decision #6: DAYS_EMPLOYED = 365243 marks a pensioner-heavy group ->
    # add FLAG_NOT_EMPLOYED before replacing the placeholder with missing data.
    cleaned["FLAG_NOT_EMPLOYED"] = (
        cleaned["DAYS_EMPLOYED"].eq(365243).astype("int8")
    )

    # EDA decision #5: DAYS_EMPLOYED = 365243 is an undocumented placeholder ->
    # set it to NA while retaining the application rows.
    cleaned.loc[
        cleaned["DAYS_EMPLOYED"].eq(365243), "DAYS_EMPLOYED"
    ] = pd.NA

    # EDA decision #7: CODE_GENDER = XNA is removed from training only; test rows
    # are retained because every test applicant must receive a prediction.
    if split == "train":
        cleaned = cleaned.loc[cleaned["CODE_GENDER"] != "XNA"].copy()

    # EDA decision #8: REGION_RATING_CLIENT_W_CITY = -1 is outside its documented
    # domain -> replace it with the mode learned from retained training data.
    cleaned.loc[
        cleaned["REGION_RATING_CLIENT_W_CITY"].eq(-1),
        "REGION_RATING_CLIENT_W_CITY",
    ] = parameters.region_rating_client_w_city_mode

    # EDA decision #21: derive affordability and loan-structure ratios.
    # EDA decision #22: non-positive denominators do not form valid ratios -> NA.
    cleaned["ANNUITY_TO_INCOME"] = _safe_ratio(
        cleaned["AMT_ANNUITY"], cleaned["AMT_INCOME_TOTAL"]
    )
    cleaned["CREDIT_TO_INCOME"] = _safe_ratio(
        cleaned["AMT_CREDIT"], cleaned["AMT_INCOME_TOTAL"]
    )
    cleaned["CREDIT_TO_GOODS_PRICE"] = _safe_ratio(
        cleaned["AMT_CREDIT"], cleaned["AMT_GOODS_PRICE"]
    )

    return cleaned


def fit_feature_engineering_parameters(
    train_df: pd.DataFrame,
    *,
    credit_amount_bins: int = 5,
) -> FeatureEngineeringParameters:
    """Learn feature-engineering settings from cleaned training data only.

    ``train_df`` should be the output of :func:`clean_application_data` with
    ``split="train"``.  In particular, the employment sentinel must already
    be missing before employment years are calculated.
    """
    _require_columns(train_df, {"AMT_CREDIT"})
    if credit_amount_bins < 2:
        raise ValueError("credit_amount_bins must be at least 2.")

    credit_amount = train_df["AMT_CREDIT"].dropna()
    if credit_amount.empty:
        raise ValueError("Cannot fit credit-amount bins without training values.")

    # EDA decision #23: loan-size risk is non-monotonic -> learn training-only
    # cut points so a later model can represent its middle-versus-tail pattern.
    quantiles = np.linspace(0, 1, credit_amount_bins + 1)
    inner_edges = np.unique(credit_amount.quantile(quantiles).to_numpy())[1:-1]
    if inner_edges.size == 0:
        raise ValueError("Cannot fit credit-amount bins: training values lack variation.")
    bin_edges = tuple(np.concatenate(([-np.inf], inner_edges, [np.inf])).tolist())

    # EDA decision #11: missingness can be predictive, especially for building
    # fields -> preserve it with indicators for available important columns.
    indicator_columns = tuple(
        column
        for column in MISSINGNESS_INDICATOR_CANDIDATES
        if column in train_df.columns
    )
    return FeatureEngineeringParameters(
        credit_amount_bin_edges=bin_edges,
        missing_indicator_columns=indicator_columns,
    )


def engineer_application_features(
    df: pd.DataFrame,
    parameters: FeatureEngineeringParameters,
) -> pd.DataFrame:
    """Add features using train-fitted settings without modifying ``df``.

    Apply this function after :func:`clean_application_data`.  The same
    ``parameters`` object must be used for train, validation, test, and future
    scoring data.
    """
    _require_columns(
        df,
        {
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
        },
    )
    engineered = df.copy()

    # DAYS_BIRTH records days before application as a negative number -> AGE_YEARS
    # is positive applicant age, which can capture life-stage risk differences.
    engineered["AGE_YEARS"] = -engineered["DAYS_BIRTH"] / 365.25

    # EDA decision #5: the employment sentinel was already converted to NA ->
    # EMPLOYMENT_YEARS is positive tenure and may proxy job stability/income security.
    engineered["EMPLOYMENT_YEARS"] = -engineered["DAYS_EMPLOYED"] / 365.25

    # EDA decision #21: credit relative to income measures requested debt burden;
    # a higher burden can make repayment more difficult.
    engineered["CREDIT_TO_INCOME"] = _safe_ratio(
        engineered["AMT_CREDIT"], engineered["AMT_INCOME_TOTAL"]
    )

    # EDA decision #21: annuity relative to income measures periodic payment burden;
    # less remaining income may increase repayment difficulty.
    engineered["ANNUITY_TO_INCOME"] = _safe_ratio(
        engineered["AMT_ANNUITY"], engineered["AMT_INCOME_TOTAL"]
    )

    # EDA decision #21: annuity relative to credit approximates repayment intensity;
    # a larger scheduled payment for the same credit can strain a household budget.
    engineered["ANNUITY_TO_CREDIT"] = _safe_ratio(
        engineered["AMT_ANNUITY"], engineered["AMT_CREDIT"]
    )

    # EDA decision #21: goods price relative to credit describes financing structure;
    # departures from a typical financed share may identify different risk profiles.
    engineered["GOODS_TO_CREDIT"] = _safe_ratio(
        engineered["AMT_GOODS_PRICE"], engineered["AMT_CREDIT"]
    )

    # EDA decision #22: all ratios use NA for a non-positive denominator rather
    # than inventing a financially meaningless value.

    # Interaction: payment burden may have a different effect for applicants with
    # short versus long employment tenure, a proxy for repayment-buffer stability.
    engineered["ANNUITY_TO_INCOME_X_EMPLOYMENT_YEARS"] = (
        engineered["ANNUITY_TO_INCOME"] * engineered["EMPLOYMENT_YEARS"]
    )

    for column in parameters.missing_indicator_columns:
        if column not in engineered.columns:
            raise KeyError(
                f"Application data is missing fitted missingness field: {column}"
            )
        # EDA decision #11: a missing value can be informative in its own right ->
        # retain that information as a binary indicator before later imputation.
        engineered[f"{column}_MISSING"] = engineered[column].isna().astype("int8")

    # EDA decision #23: AMT_CREDIT has a humped default pattern -> use the stored
    # training-only quantile boundaries to represent non-linear loan-size effects.
    engineered["AMT_CREDIT_BIN"] = pd.cut(
        engineered["AMT_CREDIT"],
        bins=parameters.credit_amount_bin_edges,
        labels=False,
        include_lowest=True,
    ).astype("Int64")

    return engineered


def aggregate_bureau(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau history to exactly one row per ``SK_ID_CURR``.

    The function assumes bureau records are available as of each current
    application date. If a future extract contains post-application records,
    filter those records to the application as-of date before calling this
    function.
    """
    _require_columns(
        bureau_df,
        {"SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", *BUREAU_NUMERIC_COLUMNS},
    )

    # EDA decision #29: bureau history is future work only after an
    # application-date-safe aggregation -> build applicant-level summaries first.
    grouped = bureau_df.groupby("SK_ID_CURR", sort=False)
    bureau_features = grouped.size().rename("BUREAU_RECORD_COUNT").to_frame()
    bureau_features["BUREAU_UNIQUE_CREDIT_COUNT"] = grouped["SK_ID_BUREAU"].nunique()

    # Credit status counts show the composition of prior obligations; more active
    # obligations may indicate greater repayment burden than closed obligations.
    bureau_features["BUREAU_ACTIVE_CREDIT_COUNT"] = grouped["CREDIT_ACTIVE"].agg(
        lambda values: values.eq("Active").sum()
    )
    bureau_features["BUREAU_CLOSED_CREDIT_COUNT"] = grouped["CREDIT_ACTIVE"].agg(
        lambda values: values.eq("Closed").sum()
    )

    # Means describe a typical prior credit, maxima retain the most severe/recent
    # observed value, and sums capture a customer's total historical exposure.
    numeric_features = grouped[list(BUREAU_NUMERIC_COLUMNS)].agg(["mean", "max", "sum"])
    numeric_features.columns = [
        f"BUREAU_{column}_{statistic.upper()}"
        for column, statistic in numeric_features.columns.to_flat_index()
    ]
    bureau_features = bureau_features.join(numeric_features)

    return bureau_features.reset_index()


def left_join_bureau_features(
    application_df: pd.DataFrame,
    bureau_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join one-row-per-applicant bureau features onto applications.

    Applicants without bureau history are retained with missing bureau-derived
    features. Their later imputation and missingness-indicator policy belongs to
    the fitted application preprocessing pipeline.
    """
    _require_columns(application_df, {"SK_ID_CURR"})
    _require_columns(bureau_features, {"SK_ID_CURR"})
    if application_df["SK_ID_CURR"].duplicated().any():
        raise ValueError("Application data must contain one row per SK_ID_CURR.")
    if bureau_features["SK_ID_CURR"].duplicated().any():
        raise ValueError("Bureau features must contain one row per SK_ID_CURR.")

    feature_columns = set(bureau_features.columns).difference({"SK_ID_CURR"})
    collisions = sorted(feature_columns.intersection(application_df.columns))
    if collisions:
        raise ValueError(
            "Bureau feature names already exist in application data: "
            f"{collisions}"
        )

    # EDA decision #29: aggregate before joining so repeated bureau-credit rows
    # cannot duplicate application rows or change the one-row-per-application grain.
    return application_df.merge(
        bureau_features,
        how="left",
        on="SK_ID_CURR",
        validate="one_to_one",
        sort=False,
    )


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    bureau_df: pd.DataFrame | None = None,
    artifacts_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, PreparationArtifacts]:
    """Fit the complete preparation pipeline on train and apply it to test.

    Parameters
    ----------
    train_df, test_df:
        Raw application tables. Training must contain ``TARGET``; test must not.
    bureau_df:
        Optional application-date-safe bureau-history table. When supplied, it
        is aggregated once and left-joined to both application tables.
    artifacts_path:
        Destination for the fitted parameters. If omitted, parameters are saved
        to ``Home-Credit-Project/models/data_preparation_parameters.pkl``.

    Returns
    -------
    prepared_train, prepared_test, artifacts
        The first two frames have identical predictor columns and order;
        ``prepared_train`` has ``TARGET`` as its final additional column.
    """
    _require_columns(train_df, {"TARGET", "SK_ID_CURR"})
    _require_columns(test_df, {"SK_ID_CURR"})
    if "TARGET" in test_df.columns:
        raise ValueError("Test data must not contain TARGET.")

    train_work = train_df.copy()
    test_work = test_df.copy()

    if bureau_df is not None:
        bureau_features = aggregate_bureau(bureau_df)
        train_work = left_join_bureau_features(train_work, bureau_features)
        test_work = left_join_bureau_features(test_work, bureau_features)

    # Fit all learned cleaning values from training applications only.
    cleaning_parameters = fit_cleaning_parameters(train_work)
    cleaned_train = clean_application_data(
        train_work, cleaning_parameters, split="train"
    )
    cleaned_test = clean_application_data(test_work, cleaning_parameters, split="test")

    # Fit bin boundaries and the missingness-indicator schema from cleaned training
    # applications only, then apply the exact same settings to the test population.
    feature_parameters = fit_feature_engineering_parameters(cleaned_train)
    engineered_train = engineer_application_features(cleaned_train, feature_parameters)
    engineered_test = engineer_application_features(cleaned_test, feature_parameters)

    # TARGET is an outcome, not a predictor -> establish the final schema from
    # training predictors and require test to contain exactly that same schema.
    feature_columns = tuple(
        column for column in engineered_train.columns if column != "TARGET"
    )
    test_only_columns = sorted(set(engineered_test.columns).difference(feature_columns))
    missing_test_columns = sorted(set(feature_columns).difference(engineered_test.columns))
    if test_only_columns or missing_test_columns:
        raise ValueError(
            "Prepared train/test predictor schemas differ. "
            f"Test-only columns: {test_only_columns}; "
            f"missing test columns: {missing_test_columns}."
        )

    prepared_train = engineered_train.loc[:, [*feature_columns, "TARGET"]].copy()
    prepared_test = engineered_test.loc[:, feature_columns].copy()

    artifacts = PreparationArtifacts(
        cleaning=cleaning_parameters,
        feature_engineering=feature_parameters,
        feature_columns=feature_columns,
    )
    save_preparation_artifacts(artifacts, artifacts_path)
    return prepared_train, prepared_test, artifacts


def save_preparation_artifacts(
    artifacts: PreparationArtifacts,
    artifacts_path: str | Path | None = None,
) -> Path:
    """Persist train-fitted preparation settings for reproducible scoring."""
    path = (
        Path(artifacts_path)
        if artifacts_path is not None
        else Path(__file__).resolve().parents[2]
        / "models"
        / "data_preparation_parameters.pkl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact_file:
        pickle.dump(artifacts, artifact_file)
    return path


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio only for positive denominator values."""
    return numerator.div(denominator.where(denominator.gt(0)))


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    """Raise a clear error when an application frame lacks required fields."""
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Application data is missing required columns: {missing}")


if __name__ == "__main__":
    # Example: run the train-fitted application pipeline on the project raw files.
    project_root = Path(__file__).resolve().parents[2]
    raw_data_directory = project_root / "data" / "raw"
    raw_train = pd.read_csv(raw_data_directory / "application_train.csv")
    raw_test = pd.read_csv(raw_data_directory / "application_test.csv")

    prepared_train, prepared_test, _ = prepare_data(raw_train, raw_test)
    predictor_columns_match = list(
        prepared_train.drop(columns="TARGET").columns
    ) == list(prepared_test.columns)

    print(f"Prepared train dimensions: {prepared_train.shape}")
    print(f"Prepared test dimensions:  {prepared_test.shape}")
    print(f"Columns match apart from TARGET: {predictor_columns_match}")
    print(f"Remaining NAs in prepared train: {int(prepared_train.isna().sum().sum())}")
    print(f"Remaining NAs in prepared test:  {int(prepared_test.isna().sum().sum())}")
