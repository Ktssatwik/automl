from typing import Any, Dict

from .base_agent import BaseAgent


class ModelSelectionAgent(BaseAgent):
    name = "model_selection"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "best_model": None,
            "model_path": None,
            "note": "Phase 3 stub: model selection/save logic will be added in Phase 5.",
        }
