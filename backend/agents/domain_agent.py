from typing import Any, Dict

from .base_agent import BaseAgent


class DomainAgent(BaseAgent):
    name = "domain_understanding"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "domain": "unknown",
            "signals": [],
            "notes": "Phase 3 stub: domain inference logic will be added in Phase 4.",
        }
