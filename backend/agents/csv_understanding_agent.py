from typing import Any, Dict

from .base_agent import BaseAgent


class CSVUnderstandingAgent(BaseAgent):
    name = "csv_understanding"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        csv_path = context.get("csv_path", "")
        return {
            "csv_path": csv_path,
            "summary": "CSV structure parsed (stub).",
            "selected_target": None,
            "reasoning": "Phase 3 stub: target detection logic will be added in Phase 4.",
        }
