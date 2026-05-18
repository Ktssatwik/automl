from typing import Any, Dict

from .base_agent import BaseAgent


class ProblemTypeAgent(BaseAgent):
    name = "problem_type_detection"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "problem_type": "unknown",
            "reasoning": "Phase 3 stub: classification/regression decision will be added in Phase 4.",
        }
