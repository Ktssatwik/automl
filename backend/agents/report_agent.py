from typing import Any, Dict

from .base_agent import BaseAgent


class ReportAgent(BaseAgent):
    name = "report_generation"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": "Pipeline report generated (stub).",
            "report": {},
        }
