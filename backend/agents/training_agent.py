from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base_agent import BaseAgent


class TrainingAgent(BaseAgent):
    name = "model_training"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        preprocessor = context.get("preprocessor")
        problem_type = context.get("problem_type")

        if X_train is None or y_train is None or preprocessor is None:
            raise ValueError("TrainingAgent requires X_train, y_train, and preprocessor.")
        train_rows = int(X_train.shape[0])

        if problem_type == "classification":
            candidates = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "decision_tree": DecisionTreeClassifier(random_state=42),
            }
            # SVC with probability can be very slow on bigger datasets.
            if train_rows <= 3000:
                candidates["svm"] = SVC(probability=True, random_state=42)
        else:
            candidates = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "ridge": Ridge(alpha=1.0),
                "lasso": Lasso(alpha=0.001),
            }

        trained_pipelines: Dict[str, Pipeline] = {}
        failed_models: Dict[str, str] = {}

        for model_name, estimator in candidates.items():
            try:
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
            raise ValueError("All candidate models failed during training.")

        context["trained_pipelines"] = trained_pipelines
        context["failed_models"] = failed_models

        return {
            "problem_type": problem_type,
            "trained_models": list(trained_pipelines.keys()),
            "failed_models": failed_models,
            "total_trained": len(trained_pipelines),
        }
