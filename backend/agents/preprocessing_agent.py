from typing import Any, Dict, List

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    name = "preprocessing"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target")
        if df is None or not target_col:
            raise ValueError("PreprocessingAgent requires dataframe and selected_target.")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        dropped_id_like: List[str] = []
        n_rows = len(X)
        for col in list(X.columns):
            col_l = col.lower()
            nunique = int(X[col].nunique(dropna=True))
            if ("id" in col_l or col_l.endswith("_id")) and nunique >= int(0.9 * max(1, n_rows)):
                dropped_id_like.append(col)
                X.drop(columns=[col], inplace=True)

        date_columns: List[str] = []
        for col in X.columns:
            if is_datetime64_any_dtype(X[col]):
                date_columns.append(col)
                continue
            if X[col].dtype == "object":
                parsed = pd.to_datetime(X[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    X[col] = parsed
                    date_columns.append(col)

        for col in date_columns:
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X.drop(columns=[col], inplace=True)

        numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if col not in numeric_features]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

        context["X"] = X
        context["y"] = y
        context["feature_columns"] = list(X.columns)
        context["numeric_features"] = numeric_features
        context["categorical_features"] = categorical_features
        context["preprocessor"] = preprocessor

        return {
            "feature_columns": list(X.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "dropped_columns": dropped_id_like,
            "date_columns_processed": date_columns,
        }
