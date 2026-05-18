from typing import Any, Dict

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class ProblemTypeAgent(BaseAgent):
    name = "problem_type_detection"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target")
        if df is None or not target_col:
            raise ValueError("ProblemTypeAgent requires dataframe and selected_target in context.")

        y = df[target_col].dropna()
        if y.empty:
            raise ValueError("Selected target column contains only null values.")

        payload = {
            "target_column": target_col,
            "target_dtype": str(df[target_col].dtype),
            "target_unique_count": int(y.nunique()),
            "target_unique_ratio": round(int(y.nunique()) / max(1, int(len(y))), 4),
            "target_sample_values": y.astype(str).head(20).tolist(),
            "instruction": "Return only classification or regression based on target behavior.",
        }
        system_prompt = (
            "You are an ML task classifier. Return strict JSON keys: "
            "problem_type (classification|regression), reasoning (string)."
        )
        llm_out = llm_service.ask_json(system_prompt, payload)

        llm_type = str(llm_out.get("problem_type", "")).strip().lower()
        if llm_type not in {"classification", "regression"}:
            raise ValueError("LLM returned invalid problem_type.")

        context["problem_type"] = llm_type

        return {
            "problem_type": llm_type,
            "target_column": target_col,
            "target_unique_count": int(y.nunique()),
            "target_unique_ratio": round(int(y.nunique()) / max(1, int(len(y))), 4),
            "reasoning": str(llm_out.get("reasoning", "")),
            "decision_mode": "llm",
        }
