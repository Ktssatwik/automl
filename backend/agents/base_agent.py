from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent logic and return structured output."""
        raise NotImplementedError
