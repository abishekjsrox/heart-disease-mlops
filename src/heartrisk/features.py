"""Feature engineering and the preprocessing pipeline shared by training and serving."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk import config


class HeartRateReserve(BaseEstimator, TransformerMixin):
    """Adds `hr_reserve_pct`: max heart rate achieved as a share of the
    age-predicted maximum (220 - age). A low share during the stress test is a
    well known warning sign, and it combines two raw columns into one signal.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        predicted_max = (220 - X["age"]).replace(0, np.nan)
        X["hr_reserve_pct"] = X["thalach"] / predicted_max
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.INPUT_FEATURES
        return np.array(base + ["hr_reserve_pct"], dtype=object)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, config.CONTINUOUS_FEATURES + config.ENGINEERED_FEATURES),
        ("bin", binary, config.BINARY_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("engineer", HeartRateReserve()), ("columns", columns)])


def build_pipeline(estimator) -> Pipeline:
    """Full model = preprocessing + estimator, saved and served as one object."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def split_xy(df: pd.DataFrame):
    return df[config.INPUT_FEATURES], df[config.TARGET]
