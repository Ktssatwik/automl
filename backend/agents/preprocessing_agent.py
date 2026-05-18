from typing import Any, Dict, List

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class PreprocessingAgent(BaseAgent):
    name = "preprocessing"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        X_train = context.get("X_train")
        X_test = context.get("X_test")
        y_train = context.get("y_train")
        y_test = context.get("y_test")

        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("PreprocessingAgent requires split datasets in context.")

        X_train = X_train.copy()
        X_test = X_test.copy()

        feature_profiles: List[Dict[str, Any]] = []
        for col in X_train.columns:
            s = X_train[col]
            profile = {
                "column": col,
                "dtype": str(s.dtype),
                "null_pct": round(float(s.isna().mean() * 100), 2),
                "unique_count": int(s.nunique(dropna=True)),
                "sample_values": s.dropna().astype(str).head(5).tolist(),
            }
            if is_numeric_dtype(s):
                profile["skewness"] = round(float(s.dropna().skew()) if s.dropna().shape[0] > 2 else 0.0, 4)
            feature_profiles.append(profile)

        payload = {
            "n_train_rows": int(X_train.shape[0]),
            "feature_profiles": feature_profiles,
            "allowed": {
                "numeric_imputer": ["median", "mean"],
                "categorical_imputer": ["most_frequent", "constant_unknown"],
                "scale_numeric": [True, False],
                "drop_columns": "list of column names",
                "date_columns": "list of column names",
                "outlier_capping": {
                    "enabled": [True, False],
                    "columns": "list of numeric columns",
                    "method": ["iqr"],
                },
            },
        }
        decision = llm_service.ask_json(
            llm_service.render_prompt("preprocessing_system.j2"),
            payload,
        )

        drop_columns = [c for c in decision.get("drop_columns", []) if c in X_train.columns]
        if drop_columns:
            X_train = X_train.drop(columns=drop_columns)
            X_test = X_test.drop(columns=[c for c in drop_columns if c in X_test.columns])

        date_columns = [c for c in decision.get("date_columns", []) if c in X_train.columns]
        for col in date_columns:
            X_train[col] = pd.to_datetime(X_train[col], errors="coerce")
            X_test[col] = pd.to_datetime(X_test[col], errors="coerce")
            for df_part in (X_train, X_test):
                df_part[f"{col}_year"] = df_part[col].dt.year
                df_part[f"{col}_month"] = df_part[col].dt.month
                df_part[f"{col}_day"] = df_part[col].dt.day
                df_part.drop(columns=[col], inplace=True)

        numeric_features = [col for col in X_train.columns if is_numeric_dtype(X_train[col])]
        categorical_features = [col for col in X_train.columns if col not in numeric_features]

        outlier_decision = decision.get("outlier_capping", {}) if isinstance(decision.get("outlier_capping", {}), dict) else {}
        outlier_enabled = bool(outlier_decision.get("enabled", False))
        outlier_columns = [c for c in outlier_decision.get("columns", []) if c in numeric_features]

        if outlier_enabled:
            for col in outlier_columns:
                s = X_train[col].dropna()
                if not s.empty:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    X_train[col] = X_train[col].clip(lower=lower, upper=upper)
                    if col in X_test.columns:
                        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

        num_imp = decision.get("numeric_imputer", "median")
        if num_imp not in {"median", "mean"}:
            raise ValueError("LLM returned invalid numeric_imputer.")

        cat_imp_raw = decision.get("categorical_imputer", "most_frequent")
        if cat_imp_raw == "constant_unknown":
            cat_imp_strategy = "constant"
            cat_fill = "Unknown"
        elif cat_imp_raw == "most_frequent":
            cat_imp_strategy = "most_frequent"
            cat_fill = None
        else:
            raise ValueError("LLM returned invalid categorical_imputer.")

        scale_numeric = bool(decision.get("scale_numeric", True))

        num_steps = [("imputer", SimpleImputer(strategy=num_imp))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=num_steps)

        cat_steps = [("imputer", SimpleImputer(strategy=cat_imp_strategy, fill_value=cat_fill))]
        # Dense encoding so downstream SMOTE can operate on transformed features.
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_steps.append(("encoder", encoder))
        categorical_pipeline = Pipeline(steps=cat_steps)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

        context["X_train"] = X_train
        context["X_test"] = X_test
        context["y_train"] = y_train
        context["y_test"] = y_test
        context["feature_columns"] = list(X_train.columns)
        context["numeric_features"] = numeric_features
        context["categorical_features"] = categorical_features
        context["preprocessor"] = preprocessor
        context["preprocessing_llm_decision"] = decision

        return {
            "feature_columns": list(X_train.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "llm_decision": decision,
            "llm_response": {
                "decision_taken": str(
                    decision.get(
                        "decision_taken",
                        "Chose preprocessing policy (imputation, scaling, feature drops/date handling, outlier policy).",
                    )
                ),
                "why": str(
                    decision.get(
                        "why",
                        "LLM used feature profiles (dtype, null %, skewness, sample values) to decide transformations.",
                    )
                ),
                "raw_decision": decision,
            },
            "decision_mode": "llm",
        }
