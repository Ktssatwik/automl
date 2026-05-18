from typing import Any, Dict

from .base_agent import BaseAgent


class SplitAgent(BaseAgent):
    name = "train_test_split"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "split_ratio": "80/20",
            "stratified": False,
            "note": "Phase 3 stub.",
        }
