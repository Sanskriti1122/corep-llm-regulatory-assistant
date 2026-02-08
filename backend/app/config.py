from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    llm_provider: str
    openai_api_key: Optional[str]
    groq_api_key: Optional[str]
    openai_model: str
    groq_model: str
    chroma_persist_directory: str

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None

        if self.groq_api_key:
            self.llm_provider = "groq"
            self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif self.openai_api_key:
            self.llm_provider = "openai"
            self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            raise RuntimeError(
                "Neither OPENAI_API_KEY nor GROQ_API_KEY is set. "
                "Please set one of them in your .env file or environment variables."
            )

        self.chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", ".chroma")


@lru_cache(maxsize=1)
def get_settings():
    return Settings()
