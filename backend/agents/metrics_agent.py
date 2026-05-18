from math import sqrt
from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .base_agent import BaseAgent


class MetricsAgent(BaseAgent):
    name = "metrics_evaluation"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trained_pipelines = context.get("trained_pipelines")
        X_test = context.get("X_test")
        y_test = context.get("y_test")
        problem_type = context.get("problem_type")

        if not trained_pipelines or X_test is None or y_test is None:
            raise ValueError("MetricsAgent requires trained_pipelines, X_test, and y_test.")

        metrics: Dict[str, Dict[str, Any]] = {}

        for model_name, pipeline in trained_pipelines.items():
            preds = pipeline.predict(X_test)
            if problem_type == "classification":
                model_metrics: Dict[str, Any] = {
                    "accuracy": round(float(accuracy_score(y_test, preds)), 6),
                    "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 6),
                    "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 6),
                    "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 6),
                    "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
                }
                unique_classes = y_test.nunique(dropna=True)
                if unique_classes == 2 and hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_test)[:, 1]
                    model_metrics["roc_auc"] = round(float(roc_auc_score(y_test, probs)), 6)
                metrics[model_name] = model_metrics
            else:
                mse = float(mean_squared_error(y_test, preds))
                metrics[model_name] = {
                    "mae": round(float(mean_absolute_error(y_test, preds)), 6),
                    "mse": round(mse, 6),
                    "rmse": round(float(sqrt(mse)), 6),
                    "r2": round(float(r2_score(y_test, preds)), 6),
                }

        if problem_type == "classification":
            ranking = sorted(metrics.keys(), key=lambda n: metrics[n]["f1"], reverse=True)
            primary_metric = "f1"
        else:
            ranking = sorted(metrics.keys(), key=lambda n: metrics[n]["r2"], reverse=True)
            primary_metric = "r2"

        context["model_metrics"] = metrics
        context["ranking"] = ranking

        return {
            "primary_metric": primary_metric,
            "metrics": metrics,
            "ranking": ranking,
            "best_model_hint": ranking[0],
        }
