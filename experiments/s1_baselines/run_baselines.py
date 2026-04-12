"""
S1 — Baseline experiments on OPS-SAT-AD.
Replicates the unsupervised subset from the 30-model benchmark.
Tracks everything to MLflow.

Usage:
    python experiments/s1_baselines/run_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.data.loader import load_opssat_features, scale_for_unsupervised
from src.evaluation.metrics import compute_metrics, metrics_table


MLFLOW_EXPERIMENT = "s1_baselines_opssat"


def run():
    X_train, y_train, X_test, y_test = load_opssat_features()
    X_train_scaled, X_test_scaled, _ = scale_for_unsupervised(X_train, y_train, X_test)

    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples  |  Anomaly rate test: {y_test.mean():.1%}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models = {
        "IsolationForest": IsolationForest(
            contamination=float(y_train.mean()), random_state=42, n_estimators=200
        ),
        "OneClassSVM": OneClassSVM(kernel="rbf", nu=0.1, gamma="scale"),
    }

    all_results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train_scaled)
            preds  = (model.predict(X_test_scaled) == -1).astype(int)
            scores = -model.score_samples(X_test_scaled)

            m = compute_metrics(y_test.values, preds, scores)
            all_results[name] = m

            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})
            mlflow.log_metrics(m)
            print(f"  {name}: F1={m['f1']:.3f}  F0.5={m['f05']:.3f}  AUC={m['auc_roc']:.3f}")

    print("\n" + "=" * 62)
    print(metrics_table(all_results).to_string())
    print("=" * 62)
    print(f"\nMLflow UI:  mlflow ui --backend-store-uri mlruns/")


if __name__ == "__main__":
    run()
