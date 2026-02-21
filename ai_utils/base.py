from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, TypeVar, Optional, List, Dict, Any, Union
from contextlib import contextmanager

T = TypeVar('T', bound=BaseModel)

class LLMClient(ABC):
    """Abstract client that wraps provider-specific clients."""
    
    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]], schema: Optional[Type[T]] = None, json_mode: bool = False) -> Union[str, T, Dict[str, Any]]:
        """Generate text, structured data (Pydantic), or raw JSON."""
        pass

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models and their capabilities."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass

class LLMManager(ABC):
    """Abstract manager to handle the lifecycle of LLM clients."""
    
    @abstractmethod
    @contextmanager
    def session(self) -> LLMClient:
        """Context manager to provide a configured LLMClient."""
        pass
