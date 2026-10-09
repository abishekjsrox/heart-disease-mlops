"""Exploratory data analysis: summary tables and figures saved to reports/figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartrisk import config  # noqa: E402

PALETTE = {0: "#3B7EA1", 1: "#C4452F"}
LABELS = {0: "No disease", 1: "Disease"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def class_balance(df, out_dir):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})",
                ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    return _save(fig, "01_class_balance.png", out_dir)


def histograms(df, out_dir):
    cols = config.CONTINUOUS_FEATURES[:5]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, cols):
        for t in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == t, col], ax=ax, bins=20,
                         color=PALETTE[t], alpha=0.55, label=LABELS[t], stat="density")
        ax.set_title(col)
        ax.set_xlabel("")
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Continuous features by outcome", y=1.0)
    return _save(fig, "02_histograms.png", out_dir)


def correlation(df, out_dir):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.75}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Pearson correlation (incl. target)")
    return _save(fig, "03_correlation_heatmap.png", out_dir)


def categorical_rates(df, out_dir):
    cols = config.CATEGORICAL_FEATURES + config.BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.flat, cols):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index()
        ax.bar(rate[col].astype(int).astype(str), rate["mean"], color="#C4452F", alpha=0.8)
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{col}: disease rate")
    axes.flat[-1].axis("off")
    return _save(fig, "04_categorical_disease_rate.png", out_dir)


def boxplots(df, out_dir):
    cols = ["age", "thalach", "oldpeak", "chol"]
    long = df[cols + [config.TARGET]].melt(id_vars=config.TARGET)
    long["outcome"] = long[config.TARGET].map(LABELS)
    g = sns.catplot(data=long, x="outcome", y="value", col="variable", kind="box",
                    sharey=False, height=3.2, aspect=0.8,
                    palette=[PALETTE[0], PALETTE[1]], hue="outcome", legend=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    path = out_dir / "05_boxplots_by_outcome.png"
    g.figure.tight_layout()
    g.figure.savefig(path, dpi=130)
    plt.close(g.figure)
    return path


def summary(df) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "class_counts": {LABELS[int(k)]: int(v)
                         for k, v in df[config.TARGET].value_counts().sort_index().items()},
        "top_correlations_with_target": df.corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET).abs().sort_values(ascending=False).head(6).round(3).to_dict(),
    }


def run(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> dict:
    paths = [f(df, out_dir) for f in
             (class_balance, histograms, correlation, categorical_rates, boxplots)]
    info = summary(df)
    info["figures"] = [p.name for p in paths]
    (out_dir.parent / "eda_summary.json").write_text(json.dumps(info, indent=2))
    return info
