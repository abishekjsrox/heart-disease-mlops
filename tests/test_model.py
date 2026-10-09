import numpy as np
import pytest

from heartrisk import config
from heartrisk.predict import RiskModel
from heartrisk.train import candidates, holdout_metrics, pick_threshold

needs_model = pytest.mark.skipif(not config.MODEL_FILE.exists(),
                                 reason="run `python -m heartrisk.train` first")


def test_candidates_include_lr_and_rf():
    names = set(candidates())
    assert {"logistic_regression", "random_forest"} <= names


def test_quick_mode_shrinks_grids():
    for _, grid in candidates(quick=True).values():
        assert all(len(v) == 1 for v in grid.values())


def test_pick_threshold_reaches_target_recall():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    p = np.array([.1, .2, .6, .35, .4, .5, .7, .8, .9, .95])
    t = pick_threshold(y, p, target_recall=0.85)
    assert ((p >= t)[y == 1]).mean() >= 0.85


def test_holdout_metrics_keys_and_range():
    m = holdout_metrics(np.array([0, 1, 1, 0]), np.array([.2, .9, .6, .4]), 0.5)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())


@needs_model
def test_saved_model_predicts_sample(sample_patient):
    out = RiskModel.load().predict([sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1


@needs_model
def test_saved_model_meets_quality_bar():
    m = RiskModel.load().meta["metrics"]
    assert m["cv_roc_auc"] >= 0.80 and m["test_recall"] >= 0.80


@needs_model
def test_model_handles_missing_optional_fields(sample_patient):
    sample_patient.update({"ca": None, "thal": None})
    assert RiskModel.load().predict([sample_patient])[0]["prediction"] in (0, 1)
