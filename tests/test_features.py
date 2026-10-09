import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from heartrisk import config, data
from heartrisk.features import HeartRateReserve, build_pipeline, build_preprocessor, split_xy


def test_heart_rate_reserve_value():
    X = pd.DataFrame({"age": [60], "thalach": [120]})
    out = HeartRateReserve().transform(X)
    assert out["hr_reserve_pct"].iloc[0] == 120 / 160


def test_preprocessor_imputes_and_has_no_nans(raw_df):
    X, _ = split_xy(data.clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    # 7 numeric (6 + engineered) + 3 binary + one-hot of 4 categoricals
    assert Xt.shape[0] == len(X) and Xt.shape[1] > 13


def test_scaled_numeric_columns_are_standardised(raw_df):
    X, _ = split_xy(data.clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    n_num = len(config.CONTINUOUS_FEATURES) + len(config.ENGINEERED_FEATURES)
    assert np.allclose(Xt[:, :n_num].mean(axis=0), 0, atol=1e-6)


def test_unseen_category_does_not_break_transform(raw_df):
    X, _ = split_xy(data.clean(raw_df))
    pre = build_preprocessor().fit(X[X["cp"] != 4])
    assert pre.transform(X).shape[0] == len(X)


def test_full_pipeline_fits_and_predicts_probabilities(raw_df):
    X, y = split_xy(data.clean(raw_df))
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()
