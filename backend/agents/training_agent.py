from typing import Any, Dict

from .base_agent import BaseAgent


class TrainingAgent(BaseAgent):
    name = "model_training"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "trained_models": [],
            "note": "Phase 3 stub: model training will be added in Phase 5.",
        }
