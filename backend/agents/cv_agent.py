from statistics import mean, pstdev
from typing import Any, Dict

from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class CVAgent(BaseAgent):
    name = "cross_validation"

    def _param_grid(self, model_name: str, problem_type: str) -> Dict[str, list]:
        if problem_type == "classification":
            grids = {
                "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
                "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
                "gradient_boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
                "decision_tree": {"model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]},
                "svm": {"model__C": [0.5, 1.0, 2.0], "model__kernel": ["rbf", "linear"]},
            }
        else:
            grids = {
                "linear_regression": {},
                "random_forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
                "gradient_boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
                "decision_tree": {"model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]},
                "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
                "lasso": {"model__alpha": [0.0001, 0.001, 0.01]},
            }
        return grids.get(model_name, {})

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trained_pipelines = context.get("trained_pipelines")
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        problem_type = context.get("problem_type")

        if not trained_pipelines or X_train is None or y_train is None:
            raise ValueError("CVAgent requires trained_pipelines, X_train, and y_train.")

        decision = llm_service.ask_json(
            llm_service.render_prompt("cv_system.j2"),
            {
                "problem_type": problem_type,
                "trained_models": list(trained_pipelines.keys()),
                "train_rows": int(X_train.shape[0]),
                "allowed": {
                    "classification_scoring": ["f1_weighted", "accuracy"],
                    "regression_scoring": ["r2", "neg_mean_squared_error"],
                    "n_splits": [3, 5],
                },
            },
        )

        n_splits = int(decision.get("n_splits", 5))
        if n_splits not in {3, 5}:
            raise ValueError("LLM returned invalid n_splits.")

        scoring = str(decision.get("scoring", "")).strip()
        if problem_type == "classification":
            if scoring not in {"f1_weighted", "accuracy"}:
                raise ValueError("LLM returned invalid classification scoring.")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            if scoring not in {"r2", "neg_mean_squared_error"}:
                raise ValueError("LLM returned invalid regression scoring.")
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        results: Dict[str, Dict[str, float]] = {}
        tuned_pipelines: Dict[str, Any] = {}
        best_params_map: Dict[str, Dict[str, Any]] = {}

        for model_name, pipeline in trained_pipelines.items():
            scores = cross_val_score(clone(pipeline), X_train, y_train, cv=splitter, scoring=scoring)
            results[model_name] = {
                "cv_mean": round(float(mean(scores)), 6),
                "cv_std": round(float(pstdev(scores)), 6),
            }

            grid = self._param_grid(model_name, problem_type)
            if grid:
                tuner = GridSearchCV(
                    estimator=clone(pipeline),
                    param_grid=grid,
                    scoring=scoring,
                    cv=splitter,
                    n_jobs=-1,
                    refit=True,
                )
                tuner.fit(X_train, y_train)
                tuned_pipelines[model_name] = tuner.best_estimator_
                best_params_map[model_name] = tuner.best_params_
                results[model_name]["tuned_cv_best"] = round(float(tuner.best_score_), 6)
            else:
                # Models without practical HP grid still get final fit on full train.
                base = clone(pipeline)
                base.fit(X_train, y_train)
                tuned_pipelines[model_name] = base
                best_params_map[model_name] = {}

        ranked = sorted(results.items(), key=lambda x: x[1].get("tuned_cv_best", x[1]["cv_mean"]), reverse=True)
        context["cv_results"] = results
        context["trained_pipelines"] = tuned_pipelines
        context["best_params_map"] = best_params_map

        return {
            "scoring": scoring,
            "n_splits": n_splits,
            "cv_results": results,
            "best_params": best_params_map,
            "ranking": [name for name, _ in ranked],
            "cv_best_model_hint": ranked[0][0],
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        f"Chose CV setup n_splits={n_splits}, scoring='{scoring}'.",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        "LLM selected validation strategy appropriate for task type and model mix.",
                    )
                ),
                "raw_decision": decision,
            },
            "decision_mode": "llm",
        }
