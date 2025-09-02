# 🧘‍♂️ ZenLLM

[![PyPI version](https://badge.fury.io/py/zenllm.svg)](https://badge.fury.io/py/zenllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

The zen, simple, and unified API for LLMs with the best developer experience: two ergonomic entry points and one consistent return type.

> Philosophy: No SDK bloat. Just requests and your API keys. Multimodal in and out. Streaming that’s easy to consume.

## ✨ What’s new (breaking change)

- Two functions: generate() for single-turn, chat() for multi-turn.
- Simple inputs for 95% cases. Escape hatch for advanced parts remains.
- Always returns a structured Response (or a ResponseStream when streaming).
- Image outputs are first-class (bytes or URLs), not lost in translation.

## 🚀 Installation

```bash
pip install zenllm
```

## 💡 Quick start

First, set your provider’s API key (e.g., `export OPENAI_API_KEY="your-key"`).

### Text-only

```python
import zenllm as llm

resp = llm.generate("Why is the sky blue?", model="gpt-4.1")
print(resp.text)
```

### Vision (single image shortcut)

```python
import zenllm as llm

resp = llm.generate(
    "What is in this photo?",
    model="gemini-2.5-pro",
    image="cheeseburger.jpg",  # path, URL, bytes, or file-like accepted
)
print(resp.text)
```

### Vision (image generation output)

Gemini can return image data inline. Save them with one call.

```python
import zenllm as llm

resp = llm.generate(
    "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme",
    model="gemini-2.5-flash-image-preview",
)
resp.save_images(prefix="banana_")  # writes banana_0.png, ...
```

### Multi-turn chat with shorthands

```python
import zenllm as llm

resp = llm.chat(
    [
      ("system", "Be concise."),
      ("user", "Describe this image in one sentence.", "cheeseburger.jpg"),
    ],
    model="claude-sonnet-4-20250514",
)
print(resp.text)
```

### Streaming with typed events

```python
import zenllm as llm

stream = llm.generate(
    "Generate an image and a short caption.",
    model="gemini-2.5-flash-image-preview",
    stream=True,
)

caption = []
for ev in stream:
    if ev.type == "text":
        caption.append(ev.text)
        print(ev.text, end="", flush=True)
    elif ev.type == "image":
        with open("out.png", "wb") as f:
            f.write(ev.bytes)
final = stream.finalize()  # Response
```

### Using OpenAI-compatible endpoints

Works with local or third-party OpenAI-compatible APIs by passing `base_url`.

```python
import zenllm as llm

# Local model (e.g., Ollama or LM Studio)
resp = llm.generate(
    "Why is the sky blue?",
    model="qwen3:30b",
    base_url="http://localhost:11434/v1",
)
print(resp.text)

# Streaming
stream = llm.generate(
    "Tell me a story.",
    model="qwen3:30b",
    base_url="http://localhost:11434/v1",
    stream=True,
)
for ev in stream:
    if ev.type == "text":
        print(ev.text, end="", flush=True)
```

## 🧱 API overview

- generate(prompt=None, *, model=..., system=None, image=None, images=None, stream=False, options=None, provider=None, base_url=None, api_key=None)
- chat(messages, *, model=..., system=None, stream=False, options=None, provider=None, base_url=None, api_key=None)

Inputs:
- prompt: str
- image: single image source (path, URL, bytes, file-like)
- images: list of image sources (same kinds)
- messages shorthands:
  - "hello"
  - ("user"|"assistant"|"system", text[, images])
  - {"role":"user","text":"...", "images":[...]}
  - {"role":"user","parts":[...]}  // escape hatch for experts
- options: normalized tuning and passthrough, e.g. {"temperature": 0.7, "max_tokens": 512}.
  These are mapped per provider where needed.

Outputs:
- Always a Response object with:
  - response.text: concatenated text
  - response.parts: normalized parts
    - {"type":"text","text":"..."}
    - {"type":"image","source":{"kind":"bytes"|"url","value":...},"mime":"image/png"}
  - response.images: convenience filtered list
  - response.finish_reason, response.usage, response.raw
  - response.save_images(dir=".", prefix="img_")
  - response.to_dict() for JSON-safe structure (bytes are base64)

Streaming:
- Returns a ResponseStream. Iterate events:
  - Text events: ev.type == "text", ev.text
  - Image events: ev.type == "image", ev.bytes, ev.mime
- Call stream.finalize() to materialize a Response from the streamed events.

## ✅ Supported Providers

| Provider   | Env Var             | Prefix     | Notes                                           | Example Models                                       |
| ---------- | ------------------- | ---------- | ----------------------------------------------- | ---------------------------------------------------- |
| Anthropic  | `ANTHROPIC_API_KEY` | `claude`   | Text + Images (input via base64)                | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| DeepSeek   | `DEEPSEEK_API_KEY`  | `deepseek` | OpenAI-compatible; image support may vary       | `deepseek-chat`, `deepseek-reasoner`                 |
| Google     | `GEMINI_API_KEY`    | `gemini`   | Text + Images (inline_data base64)              | `gemini-2.5-pro`, `gemini-2.5-flash`                 |
| OpenAI     | `OPENAI_API_KEY`    | `gpt`      | Text + Images (`image_url`, supports data URLs) | `gpt-4.1`, `gpt-4o`                                  |
| TogetherAI | `TOGETHER_API_KEY`  | `together` | OpenAI-compatible; image support may vary       | `together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |

Notes:
- For OpenAI-compatible endpoints (like local models), pass `base_url` and optional `api_key`. We’ll route via the OpenAI-compatible provider and append `/chat/completions`.
- Some third-party endpoints don’t support vision. If you pass images to an unsupported model, the upstream provider may return an error.

## 📜 License

MIT License — Copyright (c) 2025 Koen van Eijk