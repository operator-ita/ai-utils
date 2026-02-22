from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, TypeVar, Optional, List, Dict, Any, Union
from contextlib import contextmanager

T = TypeVar('T', bound=BaseModel)

class AIMessage(dict):
    """
    Assistant message that supports both dict access (for history) 
    and attribute access (for developer convenience).
    """
    def __init__(self, role: str = "assistant", content: Any = None, tool_calls: Optional[List[Dict[str, Any]]] = None):
        super().__init__(role=role, content=content)
        if tool_calls:
            self["tool_calls"] = tool_calls
        self._cached_tool_calls = None

    @property
    def role(self) -> str:
        return self["role"]
    
    @property
    def content(self) -> Any:
        return self["content"]
    
    @property
    def tool_calls(self) -> Optional[List[Any]]:
        if self._cached_tool_calls is not None:
            return self._cached_tool_calls
        
        tcs = self.get("tool_calls")
        if tcs is None:
            return None
            
        from types import SimpleNamespace
        self._cached_tool_calls = [
            SimpleNamespace(
                id=tc.get("id"),
                type=tc.get("type", "function"),
                function=SimpleNamespace(
                    name=tc.get("function", {}).get("name"),
                    arguments=tc.get("function", {}).get("arguments")
                )
            ) for tc in tcs
        ]
        return self._cached_tool_calls

class LLMClient(ABC):
    """Abstract client that wraps provider-specific clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Type[T]] = None,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Union[str, T, Dict[str, Any]]:
        """Generate text, structured data (Pydantic), raw JSON, or handle tool calls."""
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
