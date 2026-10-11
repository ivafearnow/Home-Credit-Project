"""Leakage-safe model training utilities for the Home Credit project.

All transformations and optional class resampling are part of an
``imblearn.pipeline.Pipeline``. Consequently, when the pipeline is evaluated
with cross-validation, they are fit only on the training portion of each fold.
``SK_ID_CURR`` is retained separately for submission creation and is never a
model feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
ResamplingStrategy = Literal["none", "downsample", "upsample", "SMOTE"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_prepared_data(
    data_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load prepared data and separate predictors, outcome, and scoring IDs.

    Parameters
    ----------
    data_directory : str | pathlib.Path | None, default=None
        Directory containing ``train_prepared.csv`` and ``test_prepared.csv``.
        By default, uses ``data/processed`` in the project root.

    Returns
    -------
    X, y, X_test, test_ids
        Training predictors, the binary ``TARGET`` outcome, test predictors,
        and test-set ``SK_ID_CURR`` values, respectively. IDs and ``TARGET``
        are excluded from model predictors.
    """
    directory = (
        Path(data_directory)
        if data_directory is not None
        else PROJECT_ROOT / "data" / "processed"
    )
    train = pd.read_csv(directory / "train_prepared.csv")
    test = pd.read_csv(directory / "test_prepared.csv")

    _require_columns(train, {"SK_ID_CURR", "TARGET"}, "prepared training data")
    _require_columns(test, {"SK_ID_CURR"}, "prepared test data")
    if "TARGET" in test.columns:
        raise ValueError("Prepared test data must not contain TARGET.")
    if train["SK_ID_CURR"].duplicated().any() or test["SK_ID_CURR"].duplicated().any():
        raise ValueError("Prepared data must contain one row per SK_ID_CURR.")

    X = train.drop(columns=["SK_ID_CURR", "TARGET"]).copy()
    y = train["TARGET"].copy()
    X_test = test.drop(columns="SK_ID_CURR").copy()
    test_ids = test["SK_ID_CURR"].copy()

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Prepared train and test predictor schemas do not match.")
    if y.nunique() != 2:
        raise ValueError("TARGET must contain exactly two classes for ROC AUC.")
    return X, y, X_test, test_ids


def build_preprocessor() -> ColumnTransformer:
    """Create a fold-fitted preprocessing transformer for prepared features.

    Numeric columns receive median imputation followed by standardization.
    Object, category, and boolean columns receive most-frequent imputation and
    one-hot encoding. The dtype selectors are resolved during ``fit`` inside
    the enclosing pipeline, so no imputation, scaling, or category discovery
    uses a validation fold or test data.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        An unfitted transformer intended for use in an ``imblearn`` pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )


def resample_training_data(strategy: ResamplingStrategy = "none") -> Any:
    """Return a sampler for an ``imblearn`` training pipeline.

    Parameters
    ----------
    strategy : {"none", "downsample", "upsample", "SMOTE"}, default="none"
        Class-balancing approach. ``downsample`` randomly removes majority
        records, ``upsample`` duplicates minority records, and ``SMOTE`` makes
        synthetic minority samples.

    Returns
    -------
    str or imblearn sampler
        ``"passthrough"`` for no resampling, otherwise an unfitted sampler.
        Put this result in an :class:`imblearn.pipeline.Pipeline`; the sampler
        will then act only on each training fold, never on validation data.
    """
    samplers = {
        "none": "passthrough",
        "downsample": RandomUnderSampler(random_state=RANDOM_STATE),
        "upsample": RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
    }
    try:
        return samplers[strategy]
    except KeyError as error:
        choices = ", ".join(samplers)
        raise ValueError(f"strategy must be one of: {choices}.") from error


def build_model_pipeline(
    model: BaseEstimator,
    *,
    strategy: ResamplingStrategy = "none",
) -> ImbPipeline:
    """Combine preprocessing, fold-only resampling, and an estimator.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Unfitted estimator to train after preprocessing.
    strategy : {"none", "downsample", "upsample", "SMOTE"}, default="none"
        Resampling method supplied by :func:`resample_training_data`.

    Returns
    -------
    imblearn.pipeline.Pipeline
        An unfitted, leakage-safe pipeline with steps named ``preprocessor``,
        ``sampler``, and ``model``.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("sampler", resample_training_data(strategy)),
            ("model", clone(model)),
        ]
    )


def cross_validate_auc(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
) -> tuple[float, float]:
    """Evaluate a model with stratified, leakage-safe ROC AUC folds.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        An estimator or an already configured pipeline. A plain estimator is
        automatically wrapped with :func:`build_model_pipeline` using no
        resampling. To evaluate a resampling strategy, pass a pipeline from
        :func:`build_model_pipeline`.
    X, y : pandas.DataFrame, pandas.Series
        Predictor matrix and binary outcome.
    cv : int, default=3
        Number of shuffled stratified folds.

    Returns
    -------
    mean_auc, std_auc : tuple[float, float]
        Mean and population standard deviation of validation-fold ROC AUC.
    """
    estimator = _ensure_pipeline(model)
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=folds,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
    )["test_score"]
    return float(np.mean(scores)), float(np.std(scores))


def compare_models(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
) -> pd.DataFrame:
    """Compare estimators using the same stratified ROC AUC procedure.

    Parameters
    ----------
    models : mapping[str, sklearn.base.BaseEstimator]
        Named estimators or fold-safe pipelines.
    X, y : pandas.DataFrame, pandas.Series
        Predictor matrix and binary outcome.
    cv : int, default=3
        Number of shuffled stratified folds per model.

    Returns
    -------
    pandas.DataFrame
        Tidy table with ``model``, ``mean_auc``, and ``std_auc``, ordered by
        mean AUC from highest to lowest.
    """
    if not models:
        raise ValueError("models must contain at least one named estimator.")

    results = []
    for name, model in models.items():
        mean_auc, std_auc = cross_validate_auc(model, X, y, cv=cv)
        results.append({"model": name, "mean_auc": mean_auc, "std_auc": std_auc})
    return pd.DataFrame(results).sort_values(
        "mean_auc", ascending=False, ignore_index=True
    )


def tune_model(
    model: BaseEstimator | None = None,
    param_distributions: Mapping[str, Any] | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    strategy: ResamplingStrategy = "none",
) -> RandomizedSearchCV:
    """Tune a model on a reproducible stratified 5,000-row development sample.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator | None, default=None
        Estimator to tune. The default is regularized logistic regression.
    param_distributions : mapping | None, default=None
        Randomized-search parameter distributions using pipeline names (for
        example, ``{"model__C": [0.1, 1.0]}``). Defaults are supplied only for
        the default logistic-regression model.
    X, y : pandas.DataFrame, pandas.Series | None, default=None
        Prepared training data. When both are omitted, they are loaded from
        disk with :func:`load_prepared_data`.
    strategy : {"none", "downsample", "upsample", "SMOTE"}, default="none"
        Fold-only resampling approach.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        A fitted search object with ``best_estimator_`` and ``best_params_``.
    """
    if (X is None) != (y is None):
        raise ValueError("Provide both X and y, or omit both to load prepared data.")
    if X is None and y is None:
        X, y, _, _ = load_prepared_data()

    using_default_model = model is None
    if model is None:
        model = LogisticRegression(
            max_iter=1_000, solver="liblinear", random_state=RANDOM_STATE
        )
    if param_distributions is None:
        if not using_default_model:
            raise ValueError(
                "param_distributions is required when tuning a custom model."
            )
        param_distributions = {
            "model__C": np.logspace(-4, 2, 20),
            "model__class_weight": [None, "balanced"],
        }

    sample_size = min(5_000, len(X))
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=sample_size, random_state=RANDOM_STATE
    )
    sample_indices, _ = next(splitter.split(X, y))
    X_sample = X.iloc[sample_indices]
    y_sample = y.iloc[sample_indices]

    search = RandomizedSearchCV(
        estimator=build_model_pipeline(model, strategy=strategy),
        param_distributions=dict(param_distributions),
        n_iter=20,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    return search.fit(X_sample, y_sample)


def fit_final_model(
    model: BaseEstimator | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    *,
    strategy: ResamplingStrategy = "none",
) -> ImbPipeline:
    """Fit the final leakage-safe pipeline on all prepared training rows.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator | None, default=None
        Estimator to fit. Defaults to class-weighted logistic regression.
    X, y : pandas.DataFrame, pandas.Series | None, default=None
        Prepared training data. When both are omitted, data are loaded from
        disk with :func:`load_prepared_data`.
    strategy : {"none", "downsample", "upsample", "SMOTE"}, default="none"
        Resampling applied only while fitting the pipeline.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Fitted pipeline ready for ``predict_proba`` on prepared test data.
    """
    if (X is None) != (y is None):
        raise ValueError("Provide both X and y, or omit both to load prepared data.")
    if X is None and y is None:
        X, y, _, _ = load_prepared_data()
    if model is None:
        model = LogisticRegression(
            max_iter=1_000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    pipeline = build_model_pipeline(model, strategy=strategy)
    return pipeline.fit(X, y)


def make_submission(
    fitted_model: BaseEstimator,
    X_test: pd.DataFrame | None = None,
    test_ids: pd.Series | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Score test applicants and write a Kaggle-ready submission CSV.

    Parameters
    ----------
    fitted_model : sklearn.base.BaseEstimator
        Fitted estimator or pipeline that implements ``predict_proba``.
    X_test, test_ids : pandas.DataFrame, pandas.Series | None, default=None
        Prepared test predictors and their IDs. When both are omitted, they are
        loaded from disk with :func:`load_prepared_data`.
    output_path : str | pathlib.Path | None, default=None
        Output CSV destination. Defaults to
        ``outputs/submissions/submission.csv``.

    Returns
    -------
    pathlib.Path
        The written CSV path. Its columns are ``SK_ID_CURR`` and ``TARGET``.
    """
    if (X_test is None) != (test_ids is None):
        raise ValueError("Provide both X_test and test_ids, or omit both to load data.")
    if X_test is None and test_ids is None:
        _, _, X_test, test_ids = load_prepared_data()
    if not hasattr(fitted_model, "predict_proba"):
        raise TypeError("fitted_model must implement predict_proba for ROC submission.")

    probabilities = fitted_model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": probabilities})
    path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "outputs" / "submissions" / "submission.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path


def _ensure_pipeline(model: BaseEstimator) -> BaseEstimator:
    """Wrap plain estimators while preserving user-supplied pipelines."""
    if isinstance(model, (Pipeline, ImbPipeline)):
        return clone(model)
    return build_model_pipeline(model)


def _require_columns(
    df: pd.DataFrame, required: set[str], frame_name: str
) -> None:
    """Raise a clear error when a required prepared-data column is absent."""
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")
