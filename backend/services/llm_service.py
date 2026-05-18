import json
import os
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
except ImportError:
    load_dotenv = None
    ChatGroq = None
    HumanMessage = None
    SystemMessage = None


if load_dotenv is not None:
    load_dotenv()
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class LLMService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("AUTOML_LLM_MODEL", "llama-3.3-70b-versatile")
        self.enabled = bool(self.api_key) and ChatGroq is not None
        self._client = (
            ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
            if self.enabled
            else None
        )

    def ask_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self._client is None or HumanMessage is None or SystemMessage is None:
            raise RuntimeError("LLM service is not configured. Set GROQ_API_KEY and install dependencies.")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=True)),
        ]
        response = self._client.invoke(messages)
        text = getattr(response, "content", "")
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)
        text = str(text).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM output is not valid JSON text.")

        return json.loads(text[start : end + 1])


llm_service = LLMService()
