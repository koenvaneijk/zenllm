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
- CLI model picker: when you start the CLI without --model, ZenLLM now prompts you to select a model from the provider (supports OpenAI, Groq, Anthropic, DeepSeek, Gemini, Together, X.ai, and OpenAI-compatible endpoints).

## 🚀 Installation

```bash
pip install zenllm
```

## 💡 Quick start

First, set your provider’s API key (e.g., `export OPENAI_API_KEY="your-key"`).

You can also set a default model via environment:
- export ZENLLM_DEFAULT_MODEL="gpt-4.1"

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
        if getattr(ev, "bytes", None):
            with open("out.png", "wb") as f:
                f.write(ev.bytes)
        elif getattr(ev, "url", None):
            print(f"\nImage available at: {ev.url}")
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

## 📟 CLI (terminal chat)

Run an interactive chat in your terminal:

```bash
python -m zenllm --model gpt-4o-mini
```

If you omit --model, the CLI will automatically show a model picker populated from your selected provider (OpenAI, Groq, Anthropic, DeepSeek, Gemini, Together, X.ai, or any OpenAI-compatible base_url).

Options (common ones):
- --model MODEL            Model name (defaults to ZENLLM_DEFAULT_MODEL or gpt-4.1)
- --select-model           Force the interactive model picker on startup (by default, the picker appears when you did not pass --model)
- --provider PROVIDER      Force provider (openai/gpt, gemini, claude, deepseek, together, xai, groq)
- --base-url URL           OpenAI-compatible base URL (e.g., http://localhost:11434/v1)
- --api-key KEY            Override API key for this run
- --system TEXT            System prompt for the session
- --no-stream              Disable streaming output
- --temperature FLOAT      Sampling temperature
- --top-p FLOAT            Top-p nucleus sampling
- --max-tokens INT         Limit on generated tokens
- --show-usage             Print usage dict after responses (if available)
- --show-cost              Print cost estimate after responses (if pricing is known)
- --once "PROMPT"          Send a single prompt and exit (non-interactive)

Tip:
- By default, the CLI prompts for model selection when you did not pass --model.
- For OpenAI (provider "openai" or "gpt"): during interactive selection, pressing Enter selects "gpt-5".

Interactive commands:
- /help                 Show help
- /exit | /quit | :q    Exit
- /reset                Reset conversation history
- /system TEXT          Set/replace the system prompt
- /model [NAME]         Switch model; omit NAME to select interactively
- /img PATH [PATH...]   Attach image(s) to the next user message

Examples:
```bash
# Pick a model interactively from Groq
python -m zenllm --provider groq

# Local model via OpenAI-compatible API (e.g., Ollama)
python -m zenllm --base-url http://localhost:11434/v1 --model qwen2.5:7b

# One-off question, streaming, show cost
python -m zenllm --model gpt-4o-mini --show-cost --once "Why is the sky blue?"
```

Note:
- The CLI uses the same env vars as the library (e.g., OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY, XAI_API_KEY).
- Fallback chains via ZENLLM_FALLBACK are supported by the underlying API calls.

## 📚 List models programmatically

You can query available models for each provider:

```python
import zenllm as llm

# OpenAI (or other OpenAI-compatible endpoints via base_url)
openai_models = llm.list_models(provider="openai")  # or provider=None with OPENAI_API_KEY set
print([m.id for m in openai_models][:10])

# Groq
groq_models = llm.list_models(provider="groq")

# Anthropic (Claude)
claude_models = llm.list_models(provider="claude")

# DeepSeek
deepseek_models = llm.list_models(provider="deepseek")

# Google Gemini (OpenAI-compatible list endpoint)
gemini_models = llm.list_models(provider="gemini")

# Together AI
together_models = llm.list_models(provider="together")

# X.ai (Grok)
xai_models = llm.list_models(provider="xai")

# OpenAI-compatible custom base (e.g., local)
local_models = llm.list_models(base_url="http://localhost:11434/v1")
```

Each item is a ModelInfo with fields: id, created (if integer), owned_by (if provided), and raw (the full provider response item).

## 🔁 Fallback chains (automatic provider failover)

You can define an ordered chain of providers and models. ZenLLM will try them in order and move on when a provider is down, rate-limiting, or times out. By default, we do not switch mid-stream once tokens start.

Example:
```python
import zenllm as llm
from zenllm import FallbackConfig, ProviderChoice, RetryPolicy

cfg = FallbackConfig(
    chain=[
        ProviderChoice(provider="openai",   model="gpt-4o-mini"),
        ProviderChoice(provider="xai",      model="grok-2-mini"),
        ProviderChoice(provider="together", model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ],
    retry=RetryPolicy(max_attempts=2, initial_backoff=0.5, max_backoff=4.0, timeout=30),
    allow_mid_stream_switch=False,  # recommended
)

# Single-turn
resp = llm.generate("Explain CRDTs vs OT.", fallback=cfg, options={"temperature": 0.2})
print(resp.text)

# Multi-turn
resp = llm.chat([("user", "Help me debug this error…")], fallback=cfg)
print(resp.text)

# Streaming (we only lock in a provider after the first event arrives)
stream = llm.generate("Tell me a haiku about dataclasses.", stream=True, fallback=cfg)
for ev in stream:
    if ev.type == "text":
        print(ev.text, end="")
final = stream.finalize()
```

Environment default:
- You can set a default fallback chain via `ZENLLM_FALLBACK`. Format: `provider:model,provider:model,...`
  Example:
  - `export ZENLLM_FALLBACK="openai:gpt-4o-mini,xai:grok-2-mini,together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"`
- When `fallback` is not provided to `generate/chat`, ZenLLM will use the env chain if present.

Notes:
- Per-provider overrides go in `ProviderChoice(..., options={...})`. They override call-level `options`.
- If a provider reports 400/401/403/404/422 errors, we do not retry and we move to the next provider.
- Retryable errors include 408/429/5xx and network timeouts. Exponential backoff with jitter is used.

## 💰 Cost Estimation

ZenLLM automatically estimates the cost of an API call when pricing information is available for the model used.

### After a Call (Most Common)

The `Response` object returned by `generate()` and `chat()` provides methods to access cost information. This is the simplest way to track spending.

```python
import zenllm as llm

resp = llm.generate("Why is the sky blue?", model="gpt-4.1")

# Get total cost as a float
total_cost = resp.cost()
if total_cost is not None:
    print(f"Cost: ${total_cost:.6f}")

# Get a detailed breakdown
breakdown = resp.cost_breakdown()
print(breakdown)
```
This also works in the CLI via the `--show-cost` flag.

### Programmatically Before a Call

To check model pricing before making an API call, you can import the provider class directly and use its `get_model_pricing` method. This is useful for building cost calculators or user-facing UIs.

```python
from zenllm.providers.openai import OpenAIProvider
from zenllm.providers.anthropic import AnthropicProvider

# Create provider instances
openai = OpenAIProvider()
anthropic = AnthropicProvider()

# Get pricing for a specific model
gpt_price = openai.get_model_pricing("gpt-5-mini")
# Returns {'input': 0.25, 'output': 2.0}

claude_price = anthropic.get_model_pricing("claude-haiku-3.5")
# Returns {'input': 0.8, 'output': 4.0}

if gpt_price:
    print(f"GPT-5-mini input cost: ${gpt_price['input']} / 1M tokens")
```

The method returns a dictionary with `input` and `output` prices per million tokens, or `None` if the model's pricing is not available.

## 🧱 API overview

- generate(prompt=None, *, model=..., system=None, image=None, images=None, stream=False, options=None, provider=None, base_url=None, api_key=None, fallback=None)
- chat(messages, *, model=..., system=None, stream=False, options=None, provider=None, base_url=None, api_key=None, fallback=None)
- agent(messages, *, tools=None, auto_run_tools=False, model=..., system=None, stream=False, options=None, provider=None, base_url=None, api_key=None, fallback=None)

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

Helpers (escape hatch):
- zenllm.text(value) -> {"type":"text","text": "..."}
- zenllm.image(source[, mime, detail]) -> {"type":"image","source":{"kind": "...","value": ...}, ...}

Outputs:
- Always a Response object with:
  - response.text: concatenated text
  - response.parts: normalized parts
    - {"type":"text","text":"..."}
    - {"type":"image","source":{"kind":"bytes"|"url","value":...},"mime":"image/png"}
  - response.images: convenience filtered list
  - response.finish_reason, response.usage, response.raw
  - response.save_images(dir=".", prefix="img_")
  - response.cost(prompt_chars=None, completion_chars=None): total USD cost (None if pricing unknown)
  - response.cost_breakdown(prompt_chars=None, completion_chars=None): detailed dict of pricing inputs and totals
  - response.to_dict() for JSON-safe structure (bytes are base64, kind becomes "bytes_b64")

Streaming:
- Returns a ResponseStream. Iterate events:
  - Text events: ev.type == "text", ev.text
  - Image events: ev.type == "image", either ev.bytes (with ev.mime) or ev.url
- Call stream.finalize() to materialize a Response from the streamed events.

Provider selection:
- Automatic by model prefix: gpt, gemini, claude, deepseek, together, xai, grok, groq
- Override with provider="gpt"|"openai"|"openai-compatible"|"gemini"|"claude"|"deepseek"|"together"|"xai"|"groq"
- OpenAI-compatible: pass base_url (and optional api_key) and we append /chat/completions
- Fallback chains: pass fallback=FallbackConfig(...) or set env ZENLLM_FALLBACK="provider:model,provider:model,..."

## ✅ Supported Providers

| Provider   | Env Var             | Prefix       | Notes                                           | Example Models                                       |
| ---------- | ------------------- | ------------ | ----------------------------------------------- | ---------------------------------------------------- |
| Anthropic  | `ANTHROPIC_API_KEY` | `claude`     | Text + Images (input via base64)                | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| DeepSeek   | `DEEPSEEK_API_KEY`  | `deepseek`   | OpenAI-compatible; image support may vary       | `deepseek-chat`, `deepseek-reasoner`                 |
| Google     | `GEMINI_API_KEY`    | `gemini`     | Text + Images (inline_data base64)              | `gemini-2.5-pro`, `gemini-2.5-flash`                 |
| OpenAI     | `OPENAI_API_KEY`    | `gpt`        | Text + Images (`image_url`, supports data URLs) | `gpt-4.1`, `gpt-4o`                                  |
| TogetherAI | `TOGETHER_API_KEY`  | `together`   | OpenAI-compatible; image support may vary       | `together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| Groq       | `GROQ_API_KEY`      | `groq`       | OpenAI-compatible; image support may vary       | `llama-3.1-70b-versatile`                            |
| X.ai       | `XAI_API_KEY`       | `xai`, `grok` | OpenAI-compatible; image support may vary       | `grok-code-fast-1`                                   |

Notes:
- For OpenAI-compatible endpoints (like local models), pass `base_url` and optional `api_key`. We’ll route via the OpenAI-compatible provider and append `/chat/completions`.
- Some third-party endpoints don’t support vision. If you pass images to an unsupported model, the upstream provider may return an error.
- DeepSeek and Together may not accept image URLs; prefer path/bytes/file for images with those providers.

## 🧪 Experimental: @tool decorator and agent() (preview)

Define Python functions as LLM-callable tools with a simple decorator, and pass them to the high-level agent() helper. Autorun of tools is disabled by default.

Notes:
- Current preview forwards tool definitions to the provider using an OpenAI-style schema. Automatic execution of tools on the client side (autorun loop) is intentionally off by default and will be expanded in a future release.
- Provider support for tool/function calling varies. OpenAI-compatible endpoints tend to support it; others may ignore the tools field.

Example
```python
import zenllm as llm

@llm.tool(description="Get current weather by city")
def get_weather(city: str):
    """Return current weather for a city."""
    # Implement your logic here (e.g., call a REST API)
    return {"temp_c": 21.5, "condition": "sunny"}

# Send tool definitions to the model (no automatic execution by default)
resp = llm.agent(
    messages=[("user", "What's the weather in Paris right now?")],
    tools=[get_weather],           # you can also pass a list of prebuilt dict specs
    model="gpt-4.1",
    # auto_run_tools=False is the default
)

print(resp.text)
```

Decorator signature
- @zenllm.tool(name=None, description=None, parameters=None, safe=False)
  - name: override the tool name (defaults to function name)
  - description: short description (defaults to first line of docstring)
  - parameters: JSON Schema for arguments (auto-derived from type hints if omitted)
  - safe: metadata you can use to mark read-only tools (reserved for future autorun policies)

Passing raw specs (optional)
```python
tool_spec = {
    "name": "get_weather",
    "description": "Get current weather by city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}
resp = llm.agent(
    messages=[("user", "What's the weather in Paris right now?")],
    tools=[tool_spec],             # dict specs are accepted too
    model="gpt-4.1",
)
```

Tip:
- You can also pass tools directly to chat() by building the OpenAI-style schema yourself:
  options={"tools": [{"type": "function", "function": {...}}], "tool_choice": "auto"}

Roadmap:
- Streaming tool-call events, structured JSON output helpers, and an opt-in autorun loop will land in subsequent updates.

## 🧪 Advanced examples

Manual parts with helpers:
```python
from zenllm import text, image
import zenllm as llm

msgs = [
  {"role": "user", "parts": [
    text("Describe this in one sentence."),
    image("cheeseburger.jpg", detail="high"),
  ]},
]
resp = llm.chat(msgs, model="gemini-2.5-pro")
print(resp.text)
```

Provider override:
```python
import zenllm as llm

resp = llm.generate(
  "Hello!",
  model="gpt-4.1",
  provider="openai",  # or "gpt", "openai-compatible", "gemini", "claude", "deepseek", "together", "xai", "groq"
)
print(resp.text)
```

Serialization:
```python
d = resp.to_dict()  # bytes are base64-encoded with kind "bytes_b64"
```

## 📜 License

MIT License — Copyright (c) 2025 Koen van Eijk
