from typing import Any, Dict

from .base_agent import BaseAgent

try:
    from backend.services.llm_service import llm_service
except ModuleNotFoundError:
    from services.llm_service import llm_service


class DomainAgent(BaseAgent):
    name = "domain_understanding"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_col = context.get("selected_target", "")
        if df is None:
            raise ValueError("DomainAgent requires dataframe in context.")

        payload = {
            "columns": [str(c) for c in df.columns],
            "target_column": target_col,
            "allowed_domains": [
                "insurance", "housing", "loan", "healthcare", "sales", "churn", "education", "finance", "generic"
            ],
            "instruction": "Infer one domain and provide signals plus preprocessing notes.",
        }
        system_prompt = (
            "You are a domain inference agent for tabular ML. Return strict JSON keys: "
            "domain (string), signals (array of strings), notes_for_preprocessing (string)."
        )
        llm_out = llm_service.ask_json(system_prompt, payload)

        domain = str(llm_out.get("domain", "")).strip().lower()
        allowed = {"insurance", "housing", "loan", "healthcare", "sales", "churn", "education", "finance", "generic"}
        if domain not in allowed:
            raise ValueError("LLM returned invalid domain.")

        signals = llm_out.get("signals", [])
        if not isinstance(signals, list):
            raise ValueError("LLM returned invalid signals format.")

        notes = llm_out.get("notes_for_preprocessing", "")
        if not isinstance(notes, str):
            raise ValueError("LLM returned invalid notes_for_preprocessing.")

        context["domain"] = domain
        return {
            "domain": domain,
            "signals": [str(s) for s in signals[:10]],
            "notes_for_preprocessing": notes,
            "decision_mode": "llm",
        }
