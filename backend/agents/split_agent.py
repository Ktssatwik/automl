from typing import Any, Dict

from sklearn.model_selection import train_test_split

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class SplitAgent(BaseAgent):
    name = "train_test_split"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target")
        problem_type = context.get("problem_type")

        if df is None or not target_col:
            raise ValueError("SplitAgent requires dataframe and selected_target in context.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        y = context.get("target_series_cleaned")
        if y is None:
            y = df[target_col].copy()

        # LLM-driven duplicate handling before train/test split.
        duplicate_count = int(df.duplicated().sum())
        dedup_decision = llm_service.ask_json(
            llm_service.render_prompt("dedup_system.j2"),
            {
                "row_count": int(df.shape[0]),
                "duplicate_count": duplicate_count,
                "duplicate_ratio_pct": round((duplicate_count / max(1, int(df.shape[0]))) * 100, 2),
                "instruction": "Decide whether exact duplicate rows should be dropped before splitting.",
            },
        )
        drop_duplicates = bool(dedup_decision.get("drop_duplicates", False))

        if drop_duplicates and duplicate_count > 0:
            keep_mask = ~df.duplicated(keep="first")
            df_split = df.loc[keep_mask].reset_index(drop=True)
            y = y.loc[keep_mask].reset_index(drop=True)
        else:
            df_split = df.copy().reset_index(drop=True)
            y = y.reset_index(drop=True)

        X = df_split.drop(columns=[target_col]).copy()

        payload = {
            "problem_type": problem_type,
            "row_count": int(X.shape[0]),
            "class_balance": y.value_counts(dropna=False).to_dict() if problem_type == "classification" else {},
            "allowed_test_size": [0.1, 0.15, 0.2, 0.25],
            "allowed_random_state": [42, 7, 21, 123],
        }
        decision = llm_service.ask_json(
            llm_service.render_prompt("split_system.j2"),
            payload,
        )

        test_size = float(decision.get("test_size", 0.2))
        if test_size not in {0.1, 0.15, 0.2, 0.25}:
            raise ValueError("LLM returned invalid test_size.")

        random_state = int(decision.get("random_state", 42))
        if random_state not in {42, 7, 21, 123}:
            raise ValueError("LLM returned invalid random_state.")

        use_stratify = bool(decision.get("use_stratify", False))

        stratify = None
        stratified = False
        if problem_type == "classification" and use_stratify:
            class_counts = y.value_counts(dropna=False)
            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify = y
                stratified = True

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        context["X_train"] = X_train.copy()
        context["X_test"] = X_test.copy()
        context["y_train"] = y_train
        context["y_test"] = y_test

        return {
            "split_ratio": f"{int((1-test_size)*100)}/{int(test_size*100)}",
            "random_state": random_state,
            "stratified": stratified,
            "duplicates": {
                "found": duplicate_count,
                "dropped": int(duplicate_count if drop_duplicates else 0),
                "drop_duplicates": drop_duplicates,
            },
            "dedup_llm_decision": dedup_decision,
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        f"Chose test_size={test_size}, random_state={random_state}, use_stratify={use_stratify}.",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        "LLM evaluated dataset size/class balance and selected split strategy.",
                    )
                ),
                "raw_decision": decision,
            },
            "train_shape": {"rows": int(X_train.shape[0]), "columns": int(X_train.shape[1])},
            "test_shape": {"rows": int(X_test.shape[0]), "columns": int(X_test.shape[1])},
            "decision_mode": "llm",
        }
