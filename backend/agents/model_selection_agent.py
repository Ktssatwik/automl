from datetime import datetime, timezone
from typing import Any, Dict

from .base_agent import BaseAgent

try:
    from backend.services.model_store import save_model_artifacts
    from backend.services.pipeline_state import pipeline_state_service
except ModuleNotFoundError:
    from services.model_store import save_model_artifacts
    from services.pipeline_state import pipeline_state_service


class ModelSelectionAgent(BaseAgent):
    name = "model_selection"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        job_id = context.get("job_id")
        if not job_id:
            raise ValueError("ModelSelectionAgent requires job_id in context.")

        ranking = context.get("ranking")
        trained_pipelines = context.get("trained_pipelines")
        metrics = context.get("model_metrics")
        problem_type = context.get("problem_type")

        if not ranking or not trained_pipelines or not metrics:
            raise ValueError("ModelSelectionAgent requires ranking, trained_pipelines, and model_metrics.")

        best_model_name = ranking[0]
        best_pipeline = trained_pipelines[best_model_name]

        metadata = {
            "job_id": job_id,
            "selected_target": context.get("selected_target"),
            "problem_type": problem_type,
            "domain": context.get("domain"),
            "feature_columns": context.get("feature_columns", []),
            "numeric_features": context.get("numeric_features", []),
            "categorical_features": context.get("categorical_features", []),
            "cv_results": context.get("cv_results", {}),
            "metrics": metrics,
            "ranking": ranking,
            "best_model": best_model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        saved = save_model_artifacts(job_id, best_pipeline, metadata)
        pipeline_state_service.set_paths(job_id, model_path=saved["model_path"])

        if problem_type == "classification":
            best_metric_value = metrics[best_model_name].get("f1")
            best_metric_name = "f1"
        else:
            best_metric_value = metrics[best_model_name].get("r2")
            best_metric_name = "r2"

        return {
            "best_model": best_model_name,
            "best_metric": {"name": best_metric_name, "value": best_metric_value},
            "model_path": saved["model_path"],
            "metadata_path": saved["metadata_path"],
        }
