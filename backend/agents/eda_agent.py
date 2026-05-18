from typing import Any, Dict

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class EDAAgent(BaseAgent):
    name = "eda"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target")
        if df is None:
            raise ValueError("EDAAgent requires dataframe in context.")

        rows, cols = df.shape
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        missing_summary = {
            col: {
                "null_count": int(df[col].isna().sum()),
                "null_pct": round((int(df[col].isna().sum()) / max(1, rows)) * 100, 2),
            }
            for col in df.columns
        }

        numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in numeric_cols]

        numeric_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().T.fillna(0)
            for col in desc.index:
                numeric_stats[col] = {k: float(desc.loc[col, k]) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}

        categorical_stats = {
            col: {str(k): int(v) for k, v in df[col].astype(str).value_counts(dropna=False).head(5).items()}
            for col in categorical_cols
        }

        correlation_summary = {}
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            pairs = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1 :]:
                    val = corr.loc[c1, c2]
                    if pd.notna(val):
                        pairs.append((c1, c2, float(val)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            correlation_summary = {"top_correlations": [{"col_a": a, "col_b": b, "corr": round(v, 4)} for a, b, v in pairs[:10]]}

        outlier_summary = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                outlier_summary[col] = {"outlier_count": 0, "outlier_pct": 0.0}
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((s < lower) | (s > upper)).sum()
            outlier_summary[col] = {"outlier_count": int(outliers), "outlier_pct": round((int(outliers) / max(1, len(s))) * 100, 2)}

        payload = {
            "rows": int(rows),
            "columns": int(cols),
            "target_column": target_col,
            "missing_summary": missing_summary,
            "top_correlations": correlation_summary.get("top_correlations", []),
            "outlier_summary": outlier_summary,
            "instruction": "Provide concise EDA insights and modeling cautions.",
        }
        system_prompt = llm_service.render_prompt("eda_system.j2")
        llm_out = llm_service.ask_json(system_prompt, payload)
        llm_summary = llm_out.get("llm_eda_summary")
        if not isinstance(llm_summary, str) or not llm_summary.strip():
            raise ValueError("LLM returned invalid llm_eda_summary.")

        return {
            "dataset_summary": {
                "rows": int(rows),
                "columns": int(cols),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "dtypes": dtypes,
            },
            "missing_summary": missing_summary,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "correlation_summary": correlation_summary,
            "outlier_summary": outlier_summary,
            "llm_eda_summary": llm_summary.strip(),
            "llm_response": {
                "decision_taken": str(llm_out.get("decision_taken", "Generated EDA interpretation and modeling cautions.")),
                "why": str(llm_out.get("why", llm_summary.strip())),
                "raw_decision": llm_out,
            },
            "summary_mode": "llm",
        }
