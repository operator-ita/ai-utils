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

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Type[T]] = None,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Union[str, T, Dict[str, Any]]:
        config = {}
        if schema:
            config['response_mime_type'] = 'application/json'
            config['response_schema'] = schema
        if json_mode:
            config['response_mime_type'] = 'application/json'

        if tools:
            # Transform OpenAI-style tools to Gemini format
            # OpenAI: [{"type": "function", "function": {"name": "...", "parameters": {...}}}]
            # Gemini: [{"function_declarations": [{"name": "...", "parameters": {...}}]}]
            declarations = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    declarations.append(tool["function"])
                else:
                    declarations.append(tool)
            
            if declarations:
                config['tools'] = [{"function_declarations": declarations}]

        if tool_choice:
            # Transform OpenAI tool_choice to Gemini tool_config
            if tool_choice == "auto":
                config['tool_config'] = {"function_calling_config": {"mode": "AUTO"}}
            elif tool_choice == "required":
                config['tool_config'] = {"function_calling_config": {"mode": "ANY"}}
            elif tool_choice == "none":
                config['tool_config'] = {"function_calling_config": {"mode": "NONE"}}
            elif isinstance(tool_choice, dict) and "function" in tool_choice:
                # Specific function choice
                func_name = tool_choice["function"]["name"]
                config['tool_config'] = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [func_name]
                    }
                }
            else:
                config['tool_config'] = tool_choice

        # Process roles and system instructions
        contents = []
        final_system_instruction = system_prompt

        if isinstance(prompt, str):
            contents = prompt
        elif isinstance(prompt, list):
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content")
                
                if role == "system":
                    # If both are provided, the list's system message takes precedence or is merged
                    final_system_instruction = f"{final_system_instruction}\n{content}" if final_system_instruction else content
                else:
                    # Map 'assistant' and 'model' interchangeably
                    gemini_role = "model" if role in ["assistant", "model"] else "user"
                    
                    parts = []
                    
                    # Handle text content
                    if content:
                        if isinstance(content, (dict, list)):
                            import json
                            content = json.dumps(content)
                        parts.append({"text": str(content)})
                    
                    # Handle tool calls (assistant message)
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            fc = tc.get("function")
                            if fc:
                                import json
                                args = fc.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except:
                                        pass
                                
                                parts.append({
                                    "function_call": {
                                        "name": fc["name"],
                                        "args": args
                                    }
                                })
                    
                    # Handle tool response (tool/function message)
                    if role in ["tool", "function"]:
                        gemini_role = "user" # Tool results are sent as 'user' role in some Gemini SDKs or 'function'
                        # In the new genai SDK, it might be different. Let's stick to standard parts.
                        parts.append({
                            "function_response": {
                                "name": msg.get("name") or msg.get("tool_call_id"),
                                "response": content if isinstance(content, dict) else {"result": content}
                            }
                        })

                    if parts:
                        contents.append({
                            "role": gemini_role,
                            "parts": parts
                        })
        else:
            contents = prompt
        
        if final_system_instruction:
            config['system_instruction'] = final_system_instruction
        
        try:
            res = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            # Check for tool calls
            tool_calls = []
            for part in res.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append({
                        "id": f"call_{part.function_call.name}", # Gemini doesn't always have IDs like OpenAI
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": part.function_call.args
                        }
                    })
            
            if tool_calls:
                return {"tool_calls": tool_calls}

            if schema:
                return schema.model_validate_json(res.text)
            
            if json_mode:
                import json
                return json.loads(res.text)
                
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
