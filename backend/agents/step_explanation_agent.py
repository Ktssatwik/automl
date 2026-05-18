from typing import Any, Dict, List

from .base_agent import BaseAgent


class StepExplanationAgent(BaseAgent):
    name = "step_explanation"

    def _top_items(self, items: List[Dict[str, Any]], key: str, n: int = 5) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda x: float(x.get(key, 0)), reverse=True)[:n]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        outputs = context.get("outputs", {})

        csv_out = outputs.get("csv_understanding", {})
        prob_out = outputs.get("problem_type_detection", {})
        domain_out = outputs.get("domain_understanding", {})
        eda_out = outputs.get("eda", {})
        split_out = outputs.get("train_test_split", {})
        pre_out = outputs.get("preprocessing", {})
        train_out = outputs.get("model_training", {})
        cv_out = outputs.get("cross_validation", {})
        metric_out = outputs.get("metrics_evaluation", {})
        select_out = outputs.get("model_selection", {})
        report_out = outputs.get("report_generation", {})

        missing_summary = eda_out.get("missing_summary", {})
        missing_rows = [
            {"column": col, "null_pct": vals.get("null_pct", 0), "null_count": vals.get("null_count", 0)}
            for col, vals in missing_summary.items()
        ]
        top_missing = self._top_items(missing_rows, "null_pct", 8)

        correlation_rows = eda_out.get("correlation_summary", {}).get("top_correlations", [])
        outlier_summary = eda_out.get("outlier_summary", {})
        outlier_rows = [
            {"column": col, "outlier_pct": vals.get("outlier_pct", 0), "outlier_count": vals.get("outlier_count", 0)}
            for col, vals in outlier_summary.items()
        ]
        top_outliers = self._top_items(outlier_rows, "outlier_pct", 8)

        step_explanations: List[Dict[str, Any]] = [
            {
                "step": "csv_understanding",
                "what_we_did": "Loaded CSV, profiled columns (dtype/null/unique/sample), and selected target column.",
                "key_result": {
                    "dataset_shape": csv_out.get("dataset_shape"),
                    "selected_target": csv_out.get("selected_target"),
                    "reasoning": csv_out.get("reasoning"),
                },
            },
            {
                "step": "problem_type_detection",
                "what_we_did": "Analyzed target behavior and inferred ML task type.",
                "key_result": {
                    "problem_type": prob_out.get("problem_type"),
                    "reasoning": prob_out.get("reasoning"),
                },
            },
            {
                "step": "domain_understanding",
                "what_we_did": "Inferred business domain from feature/target semantics.",
                "key_result": {
                    "domain": domain_out.get("domain"),
                    "signals": domain_out.get("signals", []),
                    "notes_for_preprocessing": domain_out.get("notes_for_preprocessing"),
                },
            },
            {
                "step": "eda",
                "what_we_did": "Computed dataset-level EDA statistics and generated an LLM interpretation.",
                "key_result": {
                    "dataset_summary": eda_out.get("dataset_summary", {}),
                    "llm_eda_summary": eda_out.get("llm_eda_summary", ""),
                },
            },
            {
                "step": "train_test_split",
                "what_we_did": "Split dataset into train and test partitions (LLM-configured split settings).",
                "key_result": split_out,
            },
            {
                "step": "preprocessing",
                "what_we_did": "Applied LLM-decided preprocessing policy (imputation, scaling, feature handling).",
                "key_result": {
                    "feature_columns": pre_out.get("feature_columns", []),
                    "numeric_features": pre_out.get("numeric_features", []),
                    "categorical_features": pre_out.get("categorical_features", []),
                    "llm_decision": pre_out.get("llm_decision", {}),
                },
            },
            {
                "step": "model_training",
                "what_we_did": "Trained LLM-selected candidate models on training data.",
                "key_result": {
                    "trained_models": train_out.get("trained_models", []),
                    "failed_models": train_out.get("failed_models", {}),
                },
            },
            {
                "step": "cross_validation",
                "what_we_did": "Ran K-fold CV and hyperparameter tuning; refit best estimator per model.",
                "key_result": {
                    "scoring": cv_out.get("scoring"),
                    "n_splits": cv_out.get("n_splits"),
                    "best_params": cv_out.get("best_params", {}),
                    "cv_best_model_hint": cv_out.get("cv_best_model_hint"),
                },
            },
            {
                "step": "metrics_evaluation",
                "what_we_did": "Evaluated tuned models on holdout test set and ranked by LLM-selected metric.",
                "key_result": {
                    "primary_metric": metric_out.get("primary_metric"),
                    "best_model_hint": metric_out.get("best_model_hint"),
                    "ranking": metric_out.get("ranking", []),
                },
            },
            {
                "step": "model_selection",
                "what_we_did": "Selected final model and persisted deployable pipeline + metadata.",
                "key_result": {
                    "best_model": select_out.get("best_model"),
                    "best_metric": select_out.get("best_metric", {}),
                    "model_path": select_out.get("model_path"),
                    "metadata_path": select_out.get("metadata_path"),
                },
            },
            {
                "step": "report_generation",
                "what_we_did": "Generated consolidated final report and summary for API/UI consumption.",
                "key_result": {
                    "summary": report_out.get("summary", ""),
                    "report_keys": list(report_out.get("report", {}).keys()),
                },
            },
        ]

        detailed_eda = {
            "dataset_summary": eda_out.get("dataset_summary", {}),
            "target_distribution": eda_out.get("target_distribution", {}),
            "top_missing_columns": top_missing,
            "top_correlations": correlation_rows,
            "top_outlier_columns": top_outliers,
            "numeric_stats": eda_out.get("numeric_stats", {}),
            "categorical_stats": eda_out.get("categorical_stats", {}),
            "llm_eda_summary": eda_out.get("llm_eda_summary", ""),
        }

        return {
            "overview": "Detailed step-by-step explanation across all pipeline stages.",
            "step_explanations": step_explanations,
            "detailed_eda_output": detailed_eda,
            "llm_response": {
                "decision_taken": "Generated detailed human-readable explanations for each stage.",
                "why": "To make all pipeline actions, decisions, and outcomes transparent.",
                "raw_decision": {},
            },
        }
