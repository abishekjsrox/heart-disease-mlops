"""Draws reports/figures/architecture.png (pure matplotlib, no extra deps)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from heartrisk import config  # noqa: E402

LANES = [  # (title, y, colour)
    ("1. Data & experimentation (local / CI)", 7.6, "#E8F1F7"),
    ("2. CI/CD - GitHub Actions", 5.05, "#FDF1E7"),
    ("3. Serving - Docker + Kubernetes", 2.5, "#EAF6EC"),
    ("4. Monitoring", 0.0, "#F3EEF8"),
]
BOXES = {
    "uci": (0.3, 7.9, "UCI repository\n(download_data.py)"),
    "clean": (2.7, 7.9, "Cleaning\nranges, '?' -> NaN,\nbinary target"),
    "eda": (5.1, 7.9, "EDA\nnotebook 01\n+ figures"),
    "train": (7.5, 7.9, "Pipeline + GridSearchCV\nLR / RF / GB, 5-fold\nthreshold @ 90% recall"),
    "mlflow": (10.2, 7.9, "MLflow\nruns, metrics, plots\nregistry: champion"),
    "push": (0.3, 5.35, "git push /\npull request"),
    "lint": (2.7, 5.35, "Job 1\nflake8"),
    "test": (5.1, 5.35, "Job 2\npytest + coverage"),
    "ctrain": (7.5, 5.35, "Job 3\ntrain + quality gate\nartifacts: model, mlruns"),
    "build": (10.2, 5.35, "Job 4\ndocker build +\nsmoke test /predict"),
    "client": (0.3, 2.8, "Client\nPOST /predict\n(JSON)"),
    "svc": (2.9, 2.8, "Service: LoadBalancer\n(+ optional Ingress)"),
    "deploy": (5.5, 2.8, "Deployment\n2 replicas, probes,\nrolling update"),
    "api": (8.0, 2.8, "FastAPI pod\n/predict  /health\n/ready  /metrics"),
    "image": (10.6, 2.8, "Docker image\nheart-risk-api\n(python 3.11 slim)"),
    "logs": (0.3, 0.3, "JSON request logs\n(stdout / kubectl logs)"),
    "prom": (5.0, 0.3, "Prometheus\nscrapes /metrics every 5s"),
    "graf": (9.4, 0.3, "Grafana dashboard\nRPS, latency p95, errors,\nprediction mix"),
}
W, H = 2.1, 1.35
ARROWS = [("uci", "clean"), ("clean", "eda"), ("eda", "train"), ("train", "mlflow"),
          ("push", "lint"), ("lint", "test"), ("test", "ctrain"), ("ctrain", "build"),
          ("client", "svc"), ("svc", "deploy"), ("deploy", "api"), ("image", "api"),
          ("prom", "graf")]


def centre(k, side):
    x, y, _ = BOXES[k]
    return {"r": (x + W, y + H / 2), "l": (x, y + H / 2), "b": (x + W / 2, y),
            "t": (x + W / 2, y + H)}[side]


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.2, 10)
    ax.axis("off")
    for title, y, colour in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y), 12.8, 2.35, boxstyle="round,pad=0.02",
                                    fc=colour, ec="#BBBBBB"))
        ax.text(0.25, y + 2.12, title, fontsize=11, weight="bold", color="#333333")
    for k, (x, y, label) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05", fc="white",
                                    ec="#3B7EA1", lw=1.4))
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=8.6)
    for a, b in ARROWS:
        same_row = BOXES[a][1] == BOXES[b][1]
        left_to_right = BOXES[a][0] < BOXES[b][0]
        s, e = (("r", "l") if left_to_right else ("l", "r")) if same_row else ("b", "t")
        ax.add_patch(FancyArrowPatch(centre(a, s), centre(b, e), arrowstyle="-|>",
                                     mutation_scale=14, color="#555555", lw=1.2))
    cross = [(("mlflow", "b"), ("build", "t"), "model.joblib"),
             (("build", "b"), ("image", "t"), "image"),
             (("api", "b"), ("prom", "t"), "scrape /metrics"),
             (("api", "b"), ("logs", "t"), "request logs")]
    for (a, sa), (b, sb), label in cross:
        p, q = centre(a, sa), centre(b, sb)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                     color="#C4452F", lw=1.3, ls="--",
                                     connectionstyle="arc3,rad=0.08"))
        ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2, label, fontsize=8,
                color="#C4452F")
    ax.set_title("Heart disease risk - MLOps architecture", fontsize=14, weight="bold")
    out = config.FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
