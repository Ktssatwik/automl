from typing import Any, Dict

from .base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    name = "preprocessing"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "preprocessing_config": {"status": "stub"},
            "feature_columns": [],
        }
