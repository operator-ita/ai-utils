import os
from google import genai
from contextlib import contextmanager
from typing import Type, TypeVar, Optional, List, Dict, Any, Union
from pydantic import BaseModel
from .base import LLMClient, LLMManager

T = TypeVar('T', bound=BaseModel)

class GeminiClient(LLMClient):
    """Unified client for both AI Studio and Vertex AI."""
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model

    def generate(self, prompt: Union[str, List[Dict[str, str]]], schema: Optional[Type[T]] = None) -> Union[str, T]:
        config = {}
        if schema:
            config['response_mime_type'] = 'application/json'
            config['response_schema'] = schema

        contents = prompt
        
        try:
            res = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            if schema:
                return schema.model_validate_json(res.text)
            return res.text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        try:
            for m in self.client.models.list():
                models.append({
                    "id": m.name,
                    "supported_actions": getattr(m, 'supported_actions', []),
                    "capabilities": getattr(m, 'supported_model_methods', [])
                })
            return models
        except Exception as e:
            raise RuntimeError(f"Failed to list Gemini models: {e}")

    def close(self):
        if self.client:
            self.client.close()

class GeminiManager(LLMManager):
    """Manager for Google AI Studio (API Key)."""
    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    @contextmanager
    def session(self) -> GeminiClient:
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable")
            
        raw_client = genai.Client(api_key=self.api_key)
        wrapper = GeminiClient(raw_client, self.model)
        try:
            yield wrapper
        finally:
            wrapper.close()

class VertexManager(LLMManager):
    """Manager for Google Cloud Vertex AI (GCP Project)."""
    def __init__(self, model: str = "gemini-2.0-flash", project: Optional[str] = None, location: str = "us-central1"):
        self.model = model
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    @contextmanager
    def session(self) -> GeminiClient:
        if not self.project:
            raise ValueError("Missing GOOGLE_CLOUD_PROJECT environment variable")
            
        raw_client = genai.Client(vertexai=True, project=self.project, location=self.location)
        wrapper = GeminiClient(raw_client, self.model)
        try:
            yield wrapper
        finally:
            wrapper.close()
