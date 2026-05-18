from datetime import datetime, timezone
from typing import Any, Dict

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
    from backend.services.model_store import save_model_artifacts
    from backend.services.pipeline_state import pipeline_state_service
except ModuleNotFoundError:
    from services.llm_service import llm_service
    from services.model_store import save_model_artifacts
    from services.pipeline_state import pipeline_state_service


class ModelSelectionAgent(BaseAgent):
    name = "model_selection"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        job_id = context.get("job_id")
        ranking = context.get("ranking")
        trained_pipelines = context.get("trained_pipelines")
        metrics = context.get("model_metrics")
        problem_type = context.get("problem_type")
        primary_metric = context.get("primary_metric")

        if not job_id:
            raise ValueError("ModelSelectionAgent requires job_id in context.")
        if not ranking or not trained_pipelines or not metrics:
            raise ValueError("ModelSelectionAgent requires ranking, trained_pipelines, and model_metrics.")

        decision = llm_service.ask_json(
            llm_service.render_prompt("model_selection_system.j2"),
            {
                "problem_type": problem_type,
                "primary_metric": primary_metric,
                "ranking": ranking,
                "cv_results": context.get("cv_results", {}),
                "test_metrics": metrics,
                "candidates": list(trained_pipelines.keys()),
                "instruction": "Choose one final best model for deployment.",
            },
        )

        best_model_name = str(decision.get("best_model", "")).strip()
        if best_model_name not in trained_pipelines:
            raise ValueError("LLM returned invalid best_model.")

        best_pipeline = trained_pipelines[best_model_name]

        metadata = {
            "job_id": job_id,
            "selected_target": context.get("selected_target"),
            "problem_type": problem_type,
            "domain": context.get("domain"),
            "feature_columns": context.get("feature_columns", []),
            "numeric_features": context.get("numeric_features", []),
            "categorical_features": context.get("categorical_features", []),
            "preprocessing_llm_decision": context.get("preprocessing_llm_decision", {}),
            "cv_results": context.get("cv_results", {}),
            "best_params_map": context.get("best_params_map", {}),
            "metrics": metrics,
            "ranking": ranking,
            "best_model": best_model_name,
            "primary_metric": primary_metric,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        saved = save_model_artifacts(job_id, best_pipeline, metadata)
        pipeline_state_service.set_paths(job_id, model_path=saved["model_path"])

        best_metric_value = metrics[best_model_name].get(primary_metric)
        cv_hint_score = context.get("cv_results", {}).get(best_model_name, {})

        return {
            "best_model": best_model_name,
            "best_metric": {"name": primary_metric, "value": best_metric_value},
            "model_path": saved["model_path"],
            "metadata_path": saved["metadata_path"],
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        f"Selected final deployment model '{best_model_name}'.",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        f"LLM chose '{best_model_name}' after comparing ranking, CV and test metrics. "
                        f"Primary metric '{primary_metric}' on test set = {best_metric_value}; "
                        f"CV summary = {cv_hint_score}.",
                    )
                ),
                "raw_decision": decision,
            },
            "decision_mode": "llm",
        }
