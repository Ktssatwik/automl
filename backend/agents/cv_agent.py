from statistics import mean, pstdev
from typing import Any, Dict

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from .base_agent import BaseAgent


class CVAgent(BaseAgent):
    name = "cross_validation"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trained_pipelines = context.get("trained_pipelines")
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        problem_type = context.get("problem_type")

        if not trained_pipelines or X_train is None or y_train is None:
            raise ValueError("CVAgent requires trained_pipelines, X_train, and y_train.")
        train_rows = int(X_train.shape[0])
        n_splits = 5 if train_rows <= 3000 else 3

        if problem_type == "classification":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scoring = "r2"

        results: Dict[str, Dict[str, float]] = {}
        for model_name, pipeline in trained_pipelines.items():
            scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring=scoring)
            results[model_name] = {
                "cv_mean": round(float(mean(scores)), 6),
                "cv_std": round(float(pstdev(scores)), 6),
            }

        ranked = sorted(results.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
        context["cv_results"] = results

        return {
            "scoring": scoring,
            "n_splits": n_splits,
            "cv_results": results,
            "ranking": [name for name, _ in ranked],
            "cv_best_model_hint": ranked[0][0],
        }
