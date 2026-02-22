# AI Lab Utils (ai-utils)

**Focus on your AI experiments, not on the boilerplate.**

Hi there! 👋 If you've ever felt overwhelmed by switching between different AI providers or getting lost in the "gritty-nitty" technical details of their specific SDKs, this project is for you.

We built this lightweight SDK to keep your code clean, consistent, and easy to read—especially useful for learning labs and rapid prototyping.

## ✨ Why use this?

- **One way to rule them all**: Use the same `.generate()` method whether you're talking to Gemini, Vertex AI, or OpenAI.
- **Unified Roles**: Don't worry about `assistant` vs `model`. Use `system`, `user`, and `assistant` everywhere.
- **Friendly to your Notebooks**: Keeps the technical clutter hidden so you can focus on the prompts and logic.
- **Save your energy**: Native Pydantic support means you get structured data back as real Python objects, no manual JSON parsing needed.
- **Safe & Sound**: Uses Python's `with` statement to make sure connections are closed properly every time.

---

## 🚀 Getting Started

### 1. Install directly from GitHub
This is the easiest way to use the SDK. It handles all dependencies automatically!

```python
!pip install git+https://github.com/operator-ita/ai-utils.git
```

### 2. Set up your workspace
Whether you're in Google Colab or working locally, setting up is a breeze. Once installed, you can import the managers directly.

#### **In Google Colab (The Secure Way)**
Click the **Secrets** (key) icon on the left to add your keys, then:
```python
import os
from google.colab import userdata

# Standard setup
os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# If you're using Google Cloud / Vertex AI
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-gcp-project"
```

---

## 🛠️ How to use it

The `.generate()` method is the heart of this SDK. It works the same way across all providers.

### 1. Simple Text with System Prompt
You can pass a simple string or a full history. Use `system_prompt` to set the AI's behavior easily.

```python
from ai_utils import GeminiManager

manager = GeminiManager(model="models/gemini-2.0-flash")

with manager.session() as llm:
    response = llm.generate(
        "Explain quantum physics", 
        system_prompt="You are a friendly teacher for 10-year-olds."
    )
    print(f"AI: {response}")
```

### 2. Flexible JSON (json_mode)
Use this when you want data but don't want to define a class. It returns a Python `dict`.

```python
with manager.session() as llm:
    # No schema needed, just set json_mode=True
    data = llm.generate(
        "Extract name and city from 'My name is John and I live in NY'", 
        json_mode=True
    )
    print(f"Name: {data['name']}, City: {data['city']}")
```

### 3. Structured Data (Pydantic)
The most robust way. Pass a Pydantic model as the `schema`, and you'll get a fully validated object back.

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

with manager.session() as llm:
    user = llm.generate("Elias is 30 years old", schema=UserInfo)
    print(f"👤 {user.name} is {user.age} years old.")
```

### 4. Multi-turn Chat (Unified Roles)
The SDK automatically translates roles between providers. Use `system`, `user`, and `assistant` everywhere.

```python
history = [
    {"role": "user", "content": "Hi, I'm David"},
    {"role": "assistant", "content": "Nice to meet you David!"},
    {"role": "user", "content": "What is my name?"}
]

with manager.session() as llm:
    print(llm.generate(history)) # "Your name is David"
```

### 5. Tool Use (Function Calling)
You can provide tools (functions) to the model. The SDK returns an `AIMessage` object that unifies the response format and allows easy access to tool calls.

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            },
            "required": ["location"],
        },
    }
}]

with manager.session() as llm:
    # 1. Get tool calls from the model
    response = llm.generate("What's the weather like in NY?", tools=tools)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Tool: {tool_call.function.name}")
        
        # Arguments are unified as a JSON string (like OpenAI) for easy parsing
        args = json.loads(tool_call.function.arguments)
        print(f"City: {args['location']}")

        # 2. Add the assistant message to history 
        # (The response object works like a dict, so you can append it directly)
        history = [{"role": "user", "content": "What's the weather like in NY?"}]
        history.append(response)

        # 3. Add the tool response to history
        history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({"weather": "sunny", "temp": "25C"})
        })

        # 4. Get final answer
        final_answer = llm.generate(history)
        print(f"AI: {final_answer}")
```

> **Note**: The SDK unifies the output format for tool calls using the `AIMessage` class. This class acts as a dictionary (compatible with message history lists) but also allows **dot notation** (e.g., `response.tool_calls[0].function.name`) for developer convenience.

The SDK also maps `tool_choice` values from OpenAI format to Gemini automatically:
- `"auto"` is mapped to Gemini's `AUTO` mode.
- `"required"` is mapped to Gemini's `ANY` mode.
- `"none"` is mapped to Gemini's `NONE` mode.
- Specific function choice (e.g., `{"function": {"name": "my_func"}}`) is also supported and translated.



### 📋 List Available Models
Curious about what models your provider offers? Here's how to check their IDs and capabilities.

```python
with manager.session() as llm:
    print(f"--- Available Models ---")
    for model in llm.list_models():
        print(f"ID: {model['id']} | Actions: {model.get('supported_actions', 'N/A')}")
```

---

## 📂 Project Roadmap
- `ai_utils/base.py`: The foundation of the library.
- `ai_utils/google_provider.py`: Everything Google (Studio & Vertex).
- `ai_utils/openai_provider.py`: Your bridge to OpenAI.
- `ai_utils/__init__.py`: The shortcut to all the tools.

## 🤝 Let's connect!
This tool is built for the community. If you have ideas for new providers or features, feel free to dive into the code!

- [Google AI Studio Docs](https://aistudio.google.com/)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Legacy Google Generative AI (Avoid)](https://github.com/google-gemini/deprecated-generative-ai-python)
- [OpenAI Python Docs](https://github.com/openai/openai-python)
