import json

import numpy as np
import pandas as pd
import pytest

from heartrisk import config


@pytest.fixture
def raw_df():
    """Small synthetic frame in the raw UCI layout (incl. bad values)."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(150, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float), "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = 5      # invalid code
    df.loc[2, "chol"] = 9999   # out of range
    return df


@pytest.fixture
def sample_patient():
    return json.loads((config.ROOT_DIR / "samples" / "high_risk.json").read_text())
