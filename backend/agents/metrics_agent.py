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

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


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
                if y_test.nunique(dropna=True) == 2 and hasattr(pipeline, "predict_proba"):
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

        decision = llm_service.ask_json(
            llm_service.render_prompt("metrics_system.j2"),
            {
                "problem_type": problem_type,
                "available_metrics": ["f1", "accuracy", "roc_auc"] if problem_type == "classification" else ["r2", "rmse", "mae"],
                "model_metrics": metrics,
                "instruction": "Pick one primary metric for ranking these models.",
            },
        )

        primary_metric = str(decision.get("primary_metric", "")).strip()
        if problem_type == "classification":
            if primary_metric not in {"f1", "accuracy", "roc_auc"}:
                raise ValueError("LLM returned invalid primary_metric for classification.")
            ranking = sorted(metrics.keys(), key=lambda n: float(metrics[n].get(primary_metric, -1e9)), reverse=True)
        else:
            if primary_metric not in {"r2", "rmse", "mae"}:
                raise ValueError("LLM returned invalid primary_metric for regression.")
            if primary_metric == "r2":
                ranking = sorted(metrics.keys(), key=lambda n: float(metrics[n].get("r2", -1e9)), reverse=True)
            else:
                ranking = sorted(metrics.keys(), key=lambda n: float(metrics[n].get(primary_metric, 1e9)))

        context["model_metrics"] = metrics
        context["ranking"] = ranking
        context["primary_metric"] = primary_metric

        return {
            "primary_metric": primary_metric,
            "metrics": metrics,
            "ranking": ranking,
            "best_model_hint": ranking[0],
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        f"Selected primary evaluation metric '{primary_metric}' for ranking.",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        "LLM chose the metric it judged most appropriate for this task and model behavior.",
                    )
                ),
                "raw_decision": decision,
            },
            "decision_mode": "llm",
        }
