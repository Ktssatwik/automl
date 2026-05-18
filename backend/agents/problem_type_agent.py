from difflib import get_close_matches
from typing import Any, Dict

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class ProblemTypeAgent(BaseAgent):
    name = "problem_type_detection"

    def _clean_class_labels(self, y):
        # Normalize textual labels to reduce accidental uniqueness from casing/spaces/typos.
        y_norm = y.astype(str).str.strip().str.lower().replace({"": "unknown"})
        counts = y_norm.value_counts(dropna=False)
        frequent_labels = [lbl for lbl, cnt in counts.items() if cnt >= 3]

        mapping: Dict[str, str] = {}
        for label in y_norm.unique():
            if label in frequent_labels:
                mapping[label] = label
                continue
            match = get_close_matches(label, frequent_labels, n=1, cutoff=0.86)
            mapping[label] = match[0] if match else label

        y_clean = y_norm.map(mapping)
        return y_clean, mapping

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target")
        if df is None or not target_col:
            raise ValueError("ProblemTypeAgent requires dataframe and selected_target in context.")

        y = df[target_col].dropna()
        if y.empty:
            raise ValueError("Selected target column contains only null values.")

        cleaned_target_mapping: Dict[str, str] = {}
        y_for_decision = y
        if y.dtype == "object":
            y_for_decision, cleaned_target_mapping = self._clean_class_labels(y)

        payload = {
            "target_column": target_col,
            "target_dtype": str(df[target_col].dtype),
            "target_unique_count": int(y_for_decision.nunique()),
            "target_unique_ratio": round(int(y_for_decision.nunique()) / max(1, int(len(y_for_decision))), 4),
            "target_sample_values": y_for_decision.astype(str).head(20).tolist(),
            "instruction": "Return only classification or regression based on target behavior.",
        }
        system_prompt = llm_service.render_prompt("problem_type_system.j2")
        llm_out = llm_service.ask_json(system_prompt, payload)

        llm_type = str(llm_out.get("problem_type", "")).strip().lower()
        if llm_type not in {"classification", "regression"}:
            raise ValueError("LLM returned invalid problem_type.")

        context["problem_type"] = llm_type

        if llm_type == "classification":
            y_all = df[target_col]
            if y_all.dtype == "object":
                y_all_norm = y_all.astype(str).str.strip().str.lower().replace({"": "unknown"})
                y_all_clean = y_all_norm.map(cleaned_target_mapping).fillna(y_all_norm)
                context["target_series_cleaned"] = y_all_clean
            else:
                context["target_series_cleaned"] = y_all

        return {
            "problem_type": llm_type,
            "target_column": target_col,
            "target_unique_count": int(y_for_decision.nunique()),
            "target_unique_ratio": round(int(y_for_decision.nunique()) / max(1, int(len(y_for_decision))), 4),
            "label_cleaning_applied": bool(cleaned_target_mapping),
            "label_mapping": cleaned_target_mapping,
            "reasoning": str(llm_out.get("reasoning", "")),
            "llm_response": {
                "decision_taken": str(llm_out.get("decision_taken", f"Classified task as '{llm_type}'.")),
                "why": str(llm_out.get("why", llm_out.get("reasoning", ""))),
                "raw_decision": llm_out,
            },
            "decision_mode": "llm",
        }
