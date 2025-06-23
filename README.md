# ZenLLM

A zen, simple, and unified API to prompt LLMs from different providers using a single function call.

**Goal:** No fancy dependencies, no SDK bloat. Just `requests` and your API keys. Easily switch between models and providers without changing your code.

## Installation

```bash
pip install .
```

## Usage

```python
from zenllm import prompt

# Simple prompt to Anthropic's Claude (default)
response = prompt("Why is the sky blue?")
print(response)

# Use Google's Gemini with a system prompt
response = prompt(
    "Why is the sky blue?",
    model="gemini-1.5-pro-latest",
    system_prompt="Reply only in French."
)
print(response)

# Stream the response from Anthropic
response_stream = prompt("Tell me a story about a robot.", model="claude-3-5-sonnet-20240620", stream=True)
for chunk in response_stream:
    print(chunk, end="", flush=True)
```

## Supported Providers

- **Anthropic Claude**: Requires `ANTHROPIC_API_KEY` environment variable.
  - Example Models: `claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`
- **Google Gemini**: Requires `GEMINI_API_KEY` environment variable.
  - Example Models: `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest`

## License

MIT License - Copyright (c) 2024 Koen van Eijk