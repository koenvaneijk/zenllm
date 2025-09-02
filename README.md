# 🧘‍♂️ ZenLLM

[![PyPI version](https://badge.fury.io/py/zenllm.svg)](https://badge.fury.io/py/zenllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

The zen, simple, and unified API to prompt LLMs from different providers using a single function call.

> Our Goal: No fancy dependencies, no SDK bloat. Just `requests` and your API keys. Easily switch between models and providers without changing your code.

## ✨ Key Features

- Unified API: A single `prompt()` function for Anthropic, Google (Gemini), OpenAI, DeepSeek, and Together.
- Multimodal: Pass images via path, bytes, file-like, or URL using tiny helpers.
- Easy Model Switching: Change models and providers with a single `model` argument.
- Lightweight: Built with only the `requests` library. No heavy SDKs to install.
- Simple Authentication: Just set your provider's API key as an environment variable.

## 🚀 Installation

```bash
pip install zenllm
```

## 💡 Usage

First, set your provider’s API key as an environment variable (e.g., `export OPENAI_API_KEY="your-key"`).

### Basic Prompt (text only)

The default model is `gpt-4.1`. You can change this by setting the `ZENLLM_DEFAULT_MODEL` environment variable.

```python
from zenllm import prompt

# Uses the default model (either the fallback or ZENLLM_DEFAULT_MODEL)
response = prompt("Why is the sky blue?")
print(response)
```

### Images and Multimodal

Use the `text()` and `image()` helpers to build content parts. `image()` accepts:
- path (str or Path)
- bytes or file-like object (with `.read()`)
- URL (http/https)

```python
from zenllm import prompt, text, image

# OpenAI (Vision)
resp = prompt(
    model="gpt-4.1-mini",
    content=[text("What is in this image?"), image("cheeseburger.jpg")]
)
print(resp)

# Gemini (Google)
resp = prompt(
    model="gemini-2.5-pro",
    content=[text("Describe this photo in one sentence."), image("cheeseburger.jpg")]
)
print(resp)

# Anthropic (Claude)
resp = prompt(
    model="claude-sonnet-4-20250514",
    content=[text("Caption this."), image("cheeseburger.jpg")]
)
print(resp)
```

You can also combine plain text and images via the `images` convenience parameter:

```python
from zenllm import prompt

resp = prompt(
    "What’s in the picture?",
    model="gemini-2.5-pro",
    images=["cheeseburger.jpg"]   # path, bytes, file-like, or URL
)
print(resp)
```

### Role-aware Messages

```python
from zenllm import prompt, text, image

resp = prompt(
    model="gemini-2.5-pro",
    messages=[
        {"role": "system", "content": [text("Be concise.")]},
        {"role": "user", "content": [
            text("Compare these."),
            image("dash-q1.png"),
            image("dash-q2.png"),
        ]},
    ],
)
print(resp)
```

### Streaming Responses

Set `stream=True` to get a generator of response chunks.

```python
from zenllm import prompt

response_stream = prompt(
    "Tell me a story about a robot.", 
    model="claude-sonnet-4-20250514", 
    stream=True
)
for chunk in response_stream:
    print(chunk, end="", flush=True)
```

### Using OpenAI-Compatible APIs

You can use `zenllm` with any OpenAI-compatible API endpoint, such as local models (e.g., via Ollama, LM Studio) or other custom services. Provide a `base_url`. `zenllm` will automatically append `/chat/completions` to this URL.

```python
from zenllm import prompt

# Example with a local model endpoint (no API key needed)
response = prompt(
    "Why is the sky blue?",
    model="qwen3:30b", 
    base_url="http://localhost:11434/v1"
)
print(response)

# Example with a custom API that requires its own key
response = prompt(
    "Why is the sky blue?",
    model="custom-model-name",
    base_url="https://api.custom-provider.com/v1",
    api_key="your-custom-api-key"
)
print(response)

# Streaming also works with compatible APIs
stream = prompt(
    "Tell me a story.",
    model="qwen3:30b", 
    base_url="http://localhost:11434/v1",
    stream=True
)
for chunk in stream:
    print(chunk, end="", flush=True)
```

## ✅ Supported Providers

| Provider   | Env Var             | Prefix     | Notes                                           | Example Models                                       |
| ---------- | ------------------- | ---------- | ----------------------------------------------- | ---------------------------------------------------- |
| Anthropic  | `ANTHROPIC_API_KEY` | `claude`   | Text + Images (inline base64)                   | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| DeepSeek   | `DEEPSEEK_API_KEY`  | `deepseek` | OpenAI-compatible; image support may vary       | `deepseek-chat`, `deepseek-reasoner`                 |
| Google     | `GEMINI_API_KEY`    | `gemini`   | Text + Images (inline_data base64)              | `gemini-2.5-pro`, `gemini-2.5-flash`                 |
| OpenAI     | `OPENAI_API_KEY`    | `gpt`      | Text + Images (`image_url`, supports data URLs) | `gpt-4.1`, `gpt-4o`                                  |
| TogetherAI | `TOGETHER_API_KEY`  | `together` | OpenAI-compatible; image support may vary       | `together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |

Note:
- For OpenAI-compatible endpoints (like local models), provide the `base_url` parameter. This will route the request correctly, regardless of the model name's prefix.
- Some third-party or OSS endpoints may not support vision. If you pass images to an unsupported model, the provider may return an error.

## 📜 License

MIT License — Copyright (c) 2025 Koen van Eijk