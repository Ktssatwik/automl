import json
import os
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    load_dotenv = None
    StrOutputParser = None
    ChatPromptTemplate = None
    ChatGroq = None
    Environment = None
    FileSystemLoader = None
    StrictUndefined = None


if load_dotenv is not None:
    load_dotenv()
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class LLMService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("AUTOML_LLM_MODEL", "llama-3.3-70b-versatile")
        self.enabled = bool(self.api_key) and ChatGroq is not None and ChatPromptTemplate is not None and StrOutputParser is not None
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")

        self._model = (
            ChatGroq(api_key=self.api_key, model=self.model, temperature=0)
            if self.enabled
            else None
        )
        self._jinja_env = (
            Environment(
                loader=FileSystemLoader(self.prompts_dir),
                undefined=StrictUndefined,
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            if Environment is not None and FileSystemLoader is not None and StrictUndefined is not None
            else None
        )

    def render_prompt(self, template_name: str, **kwargs: Any) -> str:
        if self._jinja_env is None:
            raise RuntimeError("Jinja environment is not available. Install jinja2.")
        template = self._jinja_env.get_template(template_name)
        return template.render(**kwargs).strip()

    def ask_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self._model is None or ChatPromptTemplate is None or StrOutputParser is None:
            raise RuntimeError("LLM service is not configured. Set GROQ_API_KEY and install dependencies.")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_payload}"),
            ]
        )

        chain = prompt | self._model | StrOutputParser()

        text = chain.invoke(
            {
                "system_prompt": system_prompt,
                "user_payload": json.dumps(user_payload, ensure_ascii=True),
            }
        )

        text = str(text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM output is not valid JSON text.")

        return json.loads(text[start : end + 1])


llm_service = LLMService()
