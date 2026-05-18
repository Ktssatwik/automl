from typing import Any, Dict

from .base_agent import BaseAgent


class EDAAgent(BaseAgent):
    name = "eda"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "eda_summary": "EDA completed (stub).",
            "missing_summary": {},
            "numeric_stats": {},
            "categorical_stats": {},
        }
