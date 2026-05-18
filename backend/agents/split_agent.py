from typing import Any, Dict

from sklearn.model_selection import train_test_split

from .base_agent import BaseAgent


class SplitAgent(BaseAgent):
    name = "train_test_split"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X = context.get("X")
        y = context.get("y")
        problem_type = context.get("problem_type")

        if X is None or y is None:
            raise ValueError("SplitAgent requires X and y in context.")

        stratify = None
        stratified = False
        if problem_type == "classification":
            class_counts = y.value_counts(dropna=False)
            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify = y
                stratified = True

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        context["X_train"] = X_train
        context["X_test"] = X_test
        context["y_train"] = y_train
        context["y_test"] = y_test

        return {
            "split_ratio": "80/20",
            "random_state": 42,
            "stratified": stratified,
            "train_shape": {"rows": int(X_train.shape[0]), "columns": int(X_train.shape[1])},
            "test_shape": {"rows": int(X_test.shape[0]), "columns": int(X_test.shape[1])},
        }
