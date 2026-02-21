import os
from openai import OpenAI
from contextlib import contextmanager
from typing import Type, TypeVar, Optional, List, Dict, Any, Union
from pydantic import BaseModel
from .base import LLMClient, LLMManager

T = TypeVar('T', bound=BaseModel)

class OpenAIClient(LLMClient):
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate(self, prompt: Union[str, List[Dict[str, str]]], schema: Optional[Type[T]] = None) -> Union[str, T]:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        
        try:
            if schema:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=schema,
                )
                return completion.choices[0].message.parsed
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            models = self.client.models.list()
            return [{"id": m.id, "created": m.created, "owned_by": m.owned_by} for m in models]
        except Exception as e:
            raise RuntimeError(f"Failed to list OpenAI models: {e}")

    def close(self):
        if self.client:
            self.client.close()

class OpenAIManager(LLMManager):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._raw_client = None

    @contextmanager
    def session(self) -> OpenAIClient:
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY")
            
        self._raw_client = OpenAI(api_key=self.api_key)
        wrapper = OpenAIClient(self._raw_client, self.model)
        try:
            yield wrapper
        finally:
            wrapper.close()
