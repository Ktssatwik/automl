from typing import Any, Dict

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class TrainingAgent(BaseAgent):
    name = "model_training"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        preprocessor = context.get("preprocessor")
        problem_type = context.get("problem_type")

        if X_train is None or y_train is None or preprocessor is None:
            raise ValueError("TrainingAgent requires X_train, y_train, and preprocessor.")

        if problem_type == "classification":
            all_candidates = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "decision_tree": DecisionTreeClassifier(random_state=42),
                "svm": SVC(probability=True, random_state=42),
            }
        else:
            all_candidates = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "ridge": Ridge(alpha=1.0),
                "lasso": Lasso(alpha=0.001),
            }

        decision = llm_service.ask_json(
            llm_service.render_prompt("training_system.j2"),
            {
                "problem_type": problem_type,
                "train_rows": int(X_train.shape[0]),
                "train_columns": int(X_train.shape[1]),
                "candidate_models": list(all_candidates.keys()),
                "instruction": "Choose robust models balancing speed and performance.",
            },
        )

        selected_models = [m for m in decision.get("selected_models", []) if m in all_candidates]
        if not selected_models:
            raise ValueError("LLM returned no valid selected_models.")

        trained_pipelines: Dict[str, Pipeline] = {}
        failed_models: Dict[str, str] = {}

        smote_applied = False
        class_distribution_before: Dict[str, int] = {}
        imbalance_ratio = None

        if problem_type == "classification":
            class_counts = y_train.value_counts(dropna=False)
            class_distribution_before = {str(k): int(v) for k, v in class_counts.items()}
            if len(class_counts) > 1:
                imbalance_ratio = float(class_counts.min() / max(1, class_counts.max()))
                smote_applied = imbalance_ratio < 0.8

        for model_name in selected_models:
            estimator = all_candidates[model_name]
            try:
                if problem_type == "classification" and smote_applied:
                    pipeline = ImbPipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("smote", SMOTE(random_state=42)),
                            ("model", estimator),
                        ]
                    )
                else:
                    pipeline = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("model", estimator),
                        ]
                    )
                pipeline.fit(X_train, y_train)
                trained_pipelines[model_name] = pipeline
            except Exception as exc:
                failed_models[model_name] = str(exc)

        if not trained_pipelines:
            raise ValueError("All LLM-selected candidate models failed during training.")

        context["trained_pipelines"] = trained_pipelines
        context["failed_models"] = failed_models

        return {
            "problem_type": problem_type,
            "trained_models": list(trained_pipelines.keys()),
            "failed_models": failed_models,
            "total_trained": len(trained_pipelines),
            "smote": {
                "applied": smote_applied,
                "class_distribution_before": class_distribution_before,
                "imbalance_ratio": round(float(imbalance_ratio), 4) if imbalance_ratio is not None else None,
            },
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        f"Selected models for training: {', '.join(selected_models)}.",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        "LLM balanced robustness and runtime based on dataset size and problem type. "
                        f"SMOTE {'was' if smote_applied else 'was not'} applied based on class imbalance analysis.",
                    )
                ),
                "raw_decision": decision,
            },
            "decision_mode": "llm",
        }
