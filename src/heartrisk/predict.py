"""Inference helpers shared by the API, the notebook and the CLI.

Usage:  python -m heartrisk.predict '{"age": 63, "sex": 1, ...}'
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from heartrisk import config


@dataclass
class RiskModel:
    pipeline: object
    meta: dict

    @classmethod
    def load(cls, model_path: Path = config.MODEL_FILE,
             meta_path: Path = config.MODEL_META_FILE) -> "RiskModel":
        meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        return cls(pipeline=joblib.load(model_path), meta=meta)

    @property
    def threshold(self) -> float:
        return float(self.meta.get("decision_threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.meta.get("model_version", "unknown"))

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame(records, columns=config.INPUT_FEATURES)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        out = []
        for p in proba:
            pred = int(p >= self.threshold)
            out.append({
                "prediction": pred,
                "label": "heart disease risk" if pred else "no heart disease risk",
                "probability_disease": round(float(p), 4),
                "confidence": round(float(p if pred else 1 - p), 4),
                "risk_band": "high" if p >= 0.7 else ("moderate" if pred else "low"),
                "threshold": self.threshold,
                "model_version": self.version,
            })
        return out


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.loads(
        (config.ROOT_DIR / "samples" / "high_risk.json").read_text())
    records = payload if isinstance(payload, list) else [payload]
    print(json.dumps(RiskModel.load().predict(records), indent=2))
