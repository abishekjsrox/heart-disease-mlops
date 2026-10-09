"""Train, tune and compare candidate models with MLflow tracking, then package the champion.

Usage:  python -m heartrisk.train [--quick] [--tracking-uri sqlite:///mlflow.db]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate,
                                     train_test_split)

from heartrisk import __version__, config  # noqa: E402
from heartrisk.data import load_clean, make_clean_dataset  # noqa: E402
from heartrisk.features import build_pipeline, split_xy  # noqa: E402

log = logging.getLogger("heartrisk.train")

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TARGET_RECALL = 0.90  # screening use case: missing a sick patient is the costly error
REGISTERED_NAME = "heart-disease-classifier"


def candidates(quick: bool = False) -> dict:
    """Model name -> (estimator, hyper-parameter grid)."""
    rs = config.RANDOM_STATE
    grids = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=rs),
            {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0],
             "model__penalty": ["l1", "l2"],
             "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3, 5],
             "model__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3],
             "model__subsample": [0.8, 1.0]},
        ),
    }
    if quick:  # used by CI to keep the pipeline fast
        grids = {k: (est, {p: v[:1] for p, v in g.items()}) for k, (est, g) in grids.items()}
    return grids


def pick_threshold(y_true, proba, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold whose out-of-fold recall still meets the target."""
    for t in np.round(np.arange(0.70, 0.19, -0.01), 2):
        if recall_score(y_true, (proba >= t).astype(int)) >= target_recall:
            return float(t)
    return 0.20


def holdout_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def _plots(pipe, X_test, y_test, threshold, name, out: Path) -> list[Path]:
    paths = []
    proba = pipe.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= threshold).astype(int), display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name} @ threshold {threshold:.2f}")
    fig.tight_layout()
    paths.append(out / f"confusion_matrix_{name}.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    fig.tight_layout()
    paths.append(out / f"roc_curve_{name}.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp, label = model.feature_importances_, "importance"
    else:
        imp, label = np.abs(model.coef_[0]), "|coefficient|"
    s = pd.Series(imp, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    s.plot.barh(ax=ax, color="#3B7EA1")
    ax.set_xlabel(label)
    ax.set_title(f"Top features - {name}")
    fig.tight_layout()
    paths.append(out / f"feature_importance_{name}.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.ROOT_DIR, stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:
        return os.getenv("GITHUB_SHA", "unknown")[:7]


def train(quick: bool = False, tracking_uri: str | None = None) -> dict:
    if not config.CLEAN_FILE.exists():
        make_clean_dataset()
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT_DIR / 'mlflow.db'}"))
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME,
                                 artifact_location=(config.ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    results = {}
    fig_dir = Path(tempfile.mkdtemp())
    for name, (estimator, grid) in candidates(quick).items():
        log.info("Tuning %s over %d combinations", name,
                 int(np.prod([len(v) for v in grid.values()])))
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc",
                                  cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_

            cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            cv_metrics = {f"cv_{m}": float(cv_scores[f"test_{m}"].mean()) for m in SCORING}
            cv_metrics.update({f"cv_{m}_std": float(cv_scores[f"test_{m}"].std())
                               for m in ("roc_auc", "recall")})
            oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            threshold = pick_threshold(y_train, oof)
            test_proba = best.predict_proba(X_test)[:, 1]
            test_metrics = {f"test_{k}": float(v)
                            for k, v in holdout_metrics(y_test, test_proba, threshold).items()}
            test_default = {f"test_{k}_at_0.5": float(v)
                            for k, v in holdout_metrics(y_test, test_proba, 0.5).items()
                            if k in ("precision", "recall")}

            mlflow.set_tags({"model_family": name, "quick_mode": str(quick),
                             "git_sha": _git_sha(), "dataset": "uci-heart-cleveland-303"})
            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_params({"cv_folds": config.CV_FOLDS, "test_size": config.TEST_SIZE,
                               "n_train": len(X_train), "n_test": len(X_test),
                               "grid_size": len(search.cv_results_["params"]),
                               "decision_threshold": threshold})
            mlflow.log_metrics({**cv_metrics, **test_metrics, **test_default})

            for p in _plots(best, X_test, y_test, threshold, name, fig_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            cvr = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            cvr_path = fig_dir / f"grid_results_{name}.csv"
            cvr.sort_values("rank_test_score").to_csv(cvr_path, index=False)
            mlflow.log_artifact(str(cvr_path), artifact_path="tuning")

            signature = infer_signature(X_train, best.predict_proba(X_train)[:, 1])
            model_info = mlflow.sklearn.log_model(best, name="model", signature=signature,
                                                  input_example=X_train.head(3))

            results[name] = {"pipeline": best, "run_id": run.info.run_id,
                             "model_uri": model_info.model_uri,
                             "params": search.best_params_, "threshold": threshold,
                             **cv_metrics, **test_metrics}
            log.info("%s  cv_auc=%.3f  cv_recall=%.3f  test_auc=%.3f  thr=%.2f", name,
                     cv_metrics["cv_roc_auc"], cv_metrics["cv_recall"],
                     test_metrics["test_roc_auc"], threshold)

    # champion = best mean CV ROC-AUC (selection never looks at the hold-out set)
    champion = max(results, key=lambda k: (round(results[k]["cv_roc_auc"], 3),
                                           results[k]["cv_recall"]))
    champ = results[champion]
    log.info("Champion: %s", champion)

    with mlflow.start_run(run_id=champ["run_id"]):
        mlflow.set_tag("champion", "true")
    try:
        mv = mlflow.register_model(champ["model_uri"], REGISTERED_NAME)
        mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion",
                                                         mv.version)
        registry_version = mv.version
    except Exception as exc:  # registry not available on plain file stores
        log.warning("Model registry skipped: %s", exc)
        registry_version = None

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champ["pipeline"], config.MODEL_FILE)
    meta = {
        "model_name": champion,
        "model_version": f"{__version__}+{_git_sha()}",
        "registry_name": REGISTERED_NAME,
        "registry_version": registry_version,
        "mlflow_run_id": champ["run_id"],
        "decision_threshold": champ["threshold"],
        "target_recall": TARGET_RECALL,
        "best_params": {k.replace("model__", ""): v for k, v in champ["params"].items()},
        "metrics": {k: round(v, 4) for k, v in champ.items()
                    if k.startswith(("cv_", "test_"))},
        "features": config.INPUT_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2, default=str))

    summary = pd.DataFrame({k: {m: v[m] for m in v if m.startswith(("cv_", "test_"))}
                            | {"threshold": v["threshold"]}
                            for k, v in results.items()}).T.round(4)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    for p in fig_dir.glob("*.png"):
        (config.FIGURES_DIR / p.name).write_bytes(p.read_bytes())
    print(summary.to_string())
    return meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="tiny grids (CI smoke training)")
    p.add_argument("--tracking-uri", default=None)
    args = p.parse_args()
    meta = train(quick=args.quick, tracking_uri=args.tracking_uri)
    print(json.dumps({k: meta[k] for k in ("model_name", "decision_threshold", "metrics")},
                     indent=2))


if __name__ == "__main__":
    main()
