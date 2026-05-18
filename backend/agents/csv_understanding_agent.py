from typing import Any, Dict, List

import pandas as pd

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class CSVUnderstandingAgent(BaseAgent):
    name = "csv_understanding"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        csv_path = context.get("csv_path")
        if not csv_path:
            raise ValueError("Missing csv_path in context.")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("Uploaded CSV is empty.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns (features + target).")

        n_rows = len(df)
        column_profiles: List[Dict[str, Any]] = []
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            column_profiles.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round((int(series.isna().sum()) / n_rows) * 100, 2),
                    "unique_count": int(non_null.nunique()),
                    "sample_values": non_null.astype(str).head(3).tolist(),
                }
            )

        payload = {
            "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns_profile": column_profiles,
            "instruction": "Choose exactly one most likely target column and explain briefly.",
        }
        system_prompt = llm_service.render_prompt("csv_understanding_system.j2")
        llm_out = llm_service.ask_json(system_prompt, payload)

        selected_target = llm_out.get("selected_target")
        if not isinstance(selected_target, str) or selected_target not in df.columns:
            raise ValueError("LLM returned invalid selected_target.")

        context["df"] = df
        context["selected_target"] = selected_target

        return {
            "csv_path": csv_path,
            "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns_profile": column_profiles,
            "target_candidates": [str(x) for x in llm_out.get("top_candidates", [])][:5],
            "selected_target": selected_target,
            "reasoning": str(llm_out.get("reasoning", "")),
            "llm_response": {
                "decision_taken": str(llm_out.get("decision_taken", f"Selected target column '{selected_target}'.")),
                "why": str(llm_out.get("why", llm_out.get("reasoning", ""))),
                "raw_decision": llm_out,
            },
            "decision_mode": "llm",
        }
