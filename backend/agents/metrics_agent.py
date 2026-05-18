from typing import Any, Dict

from .base_agent import BaseAgent


class MetricsAgent(BaseAgent):
    name = "metrics_evaluation"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "metrics": {},
            "ranking": [],
            "note": "Phase 3 stub: metrics logic will be added in Phase 5.",
        }
