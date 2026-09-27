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
    """Store cleaning values learned from retained training applications.

    Parameters
    ----------
    region_rating_client_w_city_mode : int | float
        Training-set mode used to repair invalid region-rating values.

    Returns
    -------
    None
        This dataclass only stores fitted values.
    """

    region_rating_client_w_city_mode: int | float


@dataclass(frozen=True)
class FeatureEngineeringParameters:
    """Store train-fitted settings for engineered application features.

    Parameters
    ----------
    credit_amount_bin_edges : tuple[float, ...]
        Training-derived boundaries for ``AMT_CREDIT`` bins.
    missing_indicator_columns : tuple[str, ...]
        Source fields for which missingness indicators are created.

    Returns
    -------
    None
        This dataclass only stores fitted values.
    """

    credit_amount_bin_edges: tuple[float, ...]
    missing_indicator_columns: tuple[str, ...]


@dataclass(frozen=True)
class PreparationArtifacts:
    """Store all artifacts needed to reproduce data preparation.

    Parameters
    ----------
    cleaning : CleaningParameters
        Fitted application-cleaning settings.
    feature_engineering : FeatureEngineeringParameters
        Fitted feature-engineering settings.
    feature_columns : tuple[str, ...]
        Final ordered predictor schema.

    Returns
    -------
    None
        This dataclass only stores fitted values.
    """

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

    Parameters
    ----------
    train_df : pandas.DataFrame
        Raw training applications used to learn cleaning values.

    Returns
    -------
    CleaningParameters
        Training-only mode used to repair invalid region-rating values.

    The returned parameters must be reused when cleaning validation, test, or
    newly scored applications. No test observations contribute to them.
    """
    _require_columns(train_df, {"CODE_GENDER", "REGION_RATING_CLIENT_W_CITY"})

    # EDA decision #1: work on a derived frame so fitting never alters the raw CSV.
    fit_frame = train_df.copy()

    # EDA decision #7: exclude the four undocumented XNA training records so
    # fitted values describe the same eligible training population as the model.
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

    # EDA decision #8: learn the replacement only from valid documented ratings,
    # preventing the invalid -1 code from affecting the repair value.
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
    df : pandas.DataFrame
        Raw application data. It is never modified in place.
    parameters : CleaningParameters
        Values returned by :func:`fit_cleaning_parameters` using training data.
    split : Split
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
    # EDA decision #1: make a derived frame so source application data stay unchanged.
    cleaned = df.copy()

    # EDA decision #6: preserve the pensioner-heavy sentinel's signal before its
    # placeholder value is removed, because dropping these applicants changes scope.
    cleaned["FLAG_NOT_EMPLOYED"] = (
        cleaned["DAYS_EMPLOYED"].eq(365243).astype("int8")
    )

    # EDA decision #5: replace the undocumented sentinel with missing data while
    # retaining applicants, since the value is not a real employment duration.
    cleaned.loc[
        cleaned["DAYS_EMPLOYED"].eq(365243), "DAYS_EMPLOYED"
    ] = pd.NA

    # EDA decision #7: remove the four undocumented training records but retain
    # test applicants, because the test population must all receive predictions.
    if split == "train":
        cleaned = cleaned.loc[cleaned["CODE_GENDER"] != "XNA"].copy()

    # EDA decision #8: repair the out-of-domain -1 code instead of deleting a test
    # applicant, using the retained-training mode to keep the replacement consistent.
    cleaned.loc[
        cleaned["REGION_RATING_CLIENT_W_CITY"].eq(-1),
        "REGION_RATING_CLIENT_W_CITY",
    ] = parameters.region_rating_client_w_city_mode

    return cleaned


def fit_feature_engineering_parameters(
    train_df: pd.DataFrame,
    *,
    credit_amount_bins: int = 5,
) -> FeatureEngineeringParameters:
    """Learn feature-engineering settings from cleaned training data only.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Cleaned training applications used to fit feature settings.
    credit_amount_bins : int, default=5
        Number of quantile-based credit-amount intervals to create.

    Returns
    -------
    FeatureEngineeringParameters
        Training-only bin boundaries and missingness-indicator schema.

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

    # EDA decision #23: fit training-only bins because loan-size risk is
    # non-monotonic and a straight-line relationship would miss its shape.
    quantiles = np.linspace(0, 1, credit_amount_bins + 1)
    inner_edges = np.unique(credit_amount.quantile(quantiles).to_numpy())[1:-1]
    if inner_edges.size == 0:
        raise ValueError("Cannot fit credit-amount bins: training values lack variation.")
    bin_edges = tuple(np.concatenate(([-np.inf], inner_edges, [np.inf])).tolist())

    # EDA decision #11: retain informative absence, especially for building data,
    # instead of allowing later imputation to erase the missingness signal.
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

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned application data to transform; it is not modified in place.
    parameters : FeatureEngineeringParameters
        Training-fitted bin boundaries and missingness-indicator schema.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with EDA-supported engineered features appended.

    Apply this function after :func:`clean_application_data`. The same
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
    # EDA decision #1: append features only to a copy so raw application values
    # remain available for audit and reproducibility.
    engineered = df.copy()

    # EDA decision #27: express the retained DAYS_BIRTH Pass candidate in years
    # so its life-stage relationship is interpretable without changing its signal.
    engineered["AGE_YEARS"] = -engineered["DAYS_BIRTH"] / 365.25

    # EDA decision #5: convert valid negative day counts to positive tenure only
    # after sentinel repair, so the placeholder cannot masquerade as long employment.
    engineered["EMPLOYMENT_YEARS"] = -engineered["DAYS_EMPLOYED"] / 365.25

    # EDA decision #21: credit relative to income represents requested debt burden,
    # which can distinguish applicants with different repayment capacity.
    engineered["CREDIT_TO_INCOME"] = _safe_ratio(
        engineered["AMT_CREDIT"], engineered["AMT_INCOME_TOTAL"]
    )

    # EDA decision #21: annuity relative to income represents periodic payment
    # burden, which can reduce the income remaining to absorb repayment shocks.
    engineered["ANNUITY_TO_INCOME"] = _safe_ratio(
        engineered["AMT_ANNUITY"], engineered["AMT_INCOME_TOTAL"]
    )

    # EDA decision #21: credit relative to goods price captures loan structure,
    # allowing the model to distinguish differently financed purchases.
    engineered["CREDIT_TO_GOODS_PRICE"] = _safe_ratio(
        engineered["AMT_CREDIT"], engineered["AMT_GOODS_PRICE"]
    )

    # EDA decision #22: _safe_ratio returns NA for non-positive denominators so
    # financially meaningless ratios are never fabricated.

    # Standard feature: interact payment burden with employment tenure because the
    # same annuity burden can imply different repayment resilience by job stability.
    engineered["ANNUITY_TO_INCOME_X_EMPLOYMENT_YEARS"] = (
        engineered["ANNUITY_TO_INCOME"] * engineered["EMPLOYMENT_YEARS"]
    )

    for column in parameters.missing_indicator_columns:
        if column not in engineered.columns:
            raise KeyError(
                f"Application data is missing fitted missingness field: {column}"
            )
        # EDA decision #11: record absence before later imputation because missing
        # values themselves can be predictive rather than merely incomplete data.
        engineered[f"{column}_MISSING"] = engineered[column].isna().astype("int8")

    # EDA decision #23: represent the humped loan-size pattern with stored
    # training-only bins rather than imposing an unsupported straight-line effect.
    engineered["AMT_CREDIT_BIN"] = pd.cut(
        engineered["AMT_CREDIT"],
        bins=parameters.credit_amount_bin_edges,
        labels=False,
        include_lowest=True,
    ).astype("Int64")

    return engineered


def aggregate_bureau(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau history to exactly one row per ``SK_ID_CURR``.

    Parameters
    ----------
    bureau_df : pandas.DataFrame
        Application-date-safe bureau records with one or more rows per applicant.

    Returns
    -------
    pandas.DataFrame
        Applicant-level bureau summaries with one row per ``SK_ID_CURR``.

    The function assumes bureau records are available as of each current
    application date. If a future extract contains post-application records,
    filter those records to the application as-of date before calling this
    function.
    """
    _require_columns(
        bureau_df,
        {"SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", *BUREAU_NUMERIC_COLUMNS},
    )

    # EDA decision #29: aggregate history to the application grain only after
    # confirming it is application-date-safe, preventing future-information leakage.
    grouped = bureau_df.groupby("SK_ID_CURR", sort=False)
    bureau_features = grouped.size().rename("BUREAU_RECORD_COUNT").to_frame()
    bureau_features["BUREAU_UNIQUE_CREDIT_COUNT"] = grouped["SK_ID_BUREAU"].nunique()

    # EDA decision #29: summarize prior-credit status at applicant level so history
    # can be used without multiplying application rows during a later safe stage.
    bureau_features["BUREAU_ACTIVE_CREDIT_COUNT"] = grouped["CREDIT_ACTIVE"].agg(
        lambda values: values.eq("Active").sum()
    )
    bureau_features["BUREAU_CLOSED_CREDIT_COUNT"] = grouped["CREDIT_ACTIVE"].agg(
        lambda values: values.eq("Closed").sum()
    )

    # EDA decision #29: create applicant-level history summaries to preserve the
    # one-row application grain required before supplementary data are modeled.
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

    Parameters
    ----------
    application_df : pandas.DataFrame
        One-row-per-applicant application data.
    bureau_features : pandas.DataFrame
        One-row-per-applicant bureau summary data.

    Returns
    -------
    pandas.DataFrame
        Application data with bureau features joined; applicants lacking history
        remain present with missing bureau-derived values.

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

    # EDA decision #29: join only aggregated history so repeated credit records
    # cannot duplicate applicants or violate the application-level study grain.
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

    # EDA decision #1: prepare copies so no raw input table is modified in place.
    train_work = train_df.copy()
    test_work = test_df.copy()

    if bureau_df is not None:
        bureau_features = aggregate_bureau(bureau_df)
        train_work = left_join_bureau_features(train_work, bureau_features)
        test_work = left_join_bureau_features(test_work, bureau_features)

    # EDA decision #8: fit the invalid-region repair on training data only so test
    # distribution information cannot influence the transformation.
    cleaning_parameters = fit_cleaning_parameters(train_work)
    cleaned_train = clean_application_data(
        train_work, cleaning_parameters, split="train"
    )
    cleaned_test = clean_application_data(test_work, cleaning_parameters, split="test")

    # EDA decision #11: fit the missingness-indicator schema on training data so
    # absence is represented consistently without test-driven feature selection.
    # EDA decision #23: fit non-linear loan-size boundaries on training data only
    # and reuse them in test, avoiding test-distribution leakage.
    feature_parameters = fit_feature_engineering_parameters(cleaned_train)
    engineered_train = engineer_application_features(cleaned_train, feature_parameters)
    engineered_test = engineer_application_features(cleaned_test, feature_parameters)

    # EDA decision #3: remove TARGET from predictors because it is the outcome,
    # not information available when scoring a new application.
    # EDA decision #4: retain SK_ID_CURR for row tracking but do not treat it as a
    # model signal; feature selection is applied downstream using this schema.
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
    """Persist train-fitted preparation settings for reproducible scoring.

    Parameters
    ----------
    artifacts : PreparationArtifacts
        Fitted cleaning, feature-engineering, and schema settings to serialize.
    artifacts_path : str | pathlib.Path | None, default=None
        Output pickle path; the project models path is used when omitted.

    Returns
    -------
    pathlib.Path
        Path of the serialized artifact file.
    """
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
    """Calculate a ratio only where its denominator is positive.

    Parameters
    ----------
    numerator : pandas.Series
        Values to divide.
    denominator : pandas.Series
        Divisor values; zero and negative values are invalid.

    Returns
    -------
    pandas.Series
        Elementwise ratio, with missing values for non-positive denominators.
    """
    return numerator.div(denominator.where(denominator.gt(0)))


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    """Validate that a DataFrame contains all required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose schema is validated.
    required : set[str]
        Column names that must be present.

    Returns
    -------
    None
        Raises ``KeyError`` when one or more required columns are absent.
    """
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Application data is missing required columns: {missing}")


if __name__ == "__main__":
    # Example: run the train-fitted application pipeline on the project raw files.
    project_root = Path(__file__).resolve().parents[2]
    raw_data_directory = project_root / "data" / "raw"
    raw_train = pd.read_csv(raw_data_directory / "application_train.csv")
    raw_test = pd.read_csv(raw_data_directory / "application_test.csv")
    # EDA decision #29: include history only through an applicant-level aggregate
    # so bureau rows cannot change the one-row-per-application population.
    raw_bureau = pd.read_csv(raw_data_directory / "bureau.csv")

    prepared_train, prepared_test, _ = prepare_data(
        raw_train, raw_test, bureau_df=raw_bureau
    )
    predictor_columns_match = list(
        prepared_train.drop(columns="TARGET").columns
    ) == list(prepared_test.columns)
    train_id_is_unique = prepared_train["SK_ID_CURR"].is_unique
    test_id_is_unique = prepared_test["SK_ID_CURR"].is_unique

    processed_data_directory = project_root / "data" / "processed"
    processed_data_directory.mkdir(parents=True, exist_ok=True)
    prepared_train.to_csv(processed_data_directory / "train_prepared.csv", index=False)
    prepared_test.to_csv(processed_data_directory / "test_prepared.csv", index=False)

    print(f"Prepared train dimensions: {prepared_train.shape}")
    print(f"Prepared test dimensions:  {prepared_test.shape}")
    print(f"Columns match apart from TARGET: {predictor_columns_match}")
    print(f"SK_ID_CURR is unique in prepared train: {train_id_is_unique}")
    print(f"SK_ID_CURR is unique in prepared test:  {test_id_is_unique}")
