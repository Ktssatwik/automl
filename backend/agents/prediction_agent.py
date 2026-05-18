from typing import Any, Dict

from .base_agent import BaseAgent


class PredictionAgent(BaseAgent):
    name = "prediction"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prediction": None,
            "note": "Phase 3 stub: online prediction logic will be added in Phase 6.",
        }
