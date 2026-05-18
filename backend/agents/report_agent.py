from datetime import datetime, timezone
from typing import Any, Dict

from .base_agent import BaseAgent


class ReportAgent(BaseAgent):
    name = "report_generation"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        outputs = context.get("outputs", {})
        csv_info = outputs.get("csv_understanding", {})
        problem_info = outputs.get("problem_type_detection", {})
        domain_info = outputs.get("domain_understanding", {})
        eda_info = outputs.get("eda", {})
        pre_info = outputs.get("preprocessing", {})
        train_info = outputs.get("model_training", {})
        cv_info = outputs.get("cross_validation", {})
        metric_info = outputs.get("metrics_evaluation", {})
        selection_info = outputs.get("model_selection", {})

        final_report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job_id": context.get("job_id"),
            "dataset_summary": csv_info.get("dataset_shape", {}),
            "selected_target": csv_info.get("selected_target"),
            "problem_type": problem_info.get("problem_type"),
            "domain": domain_info.get("domain"),
            "eda_summary": eda_info.get("llm_eda_summary", ""),
            "preprocessing": {
                "feature_count": len(pre_info.get("feature_columns", [])),
                "numeric_features": pre_info.get("numeric_features", []),
                "categorical_features": pre_info.get("categorical_features", []),
            },
            "models_trained": train_info.get("trained_models", []),
            "cv_results": cv_info.get("cv_results", {}),
            "metrics": metric_info.get("metrics", {}),
            "ranking": metric_info.get("ranking", []),
            "best_model": selection_info.get("best_model"),
            "best_metric": selection_info.get("best_metric", {}),
            "model_path": selection_info.get("model_path"),
            "metadata_path": selection_info.get("metadata_path"),
        }

        summary = (
            f"AutoML completed for job {context.get('job_id')}. "
            f"Target: {final_report.get('selected_target')}, "
            f"Task: {final_report.get('problem_type')}, "
            f"Best model: {final_report.get('best_model')}."
        )

        return {
            "summary": summary,
            "report": final_report,
            "llm_response": {
                "decision_taken": "Compiled final report from all completed pipeline stages.",
                "why": "This stage aggregates prior agent outputs into a single readable artifact.",
                "raw_decision": {},
            },
        }
