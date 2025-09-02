import os
import warnings
from typing import Any, Dict, List, Optional, Union

from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.deepseek import DeepseekProvider
from .providers.together import TogetherProvider

# A mapping from model name prefixes to provider instances
_PROVIDERS = {
    "claude": AnthropicProvider(),
    "gemini": GoogleProvider(),
    "gpt": OpenAIProvider(),
    "deepseek": DeepseekProvider(),
    "together": TogetherProvider(),
}

def _get_provider(model_name, **kwargs):
    """Selects the provider based on the model name or kwargs."""
    # If base_url is provided, always use the OpenAI provider for compatibility.
    if "base_url" in kwargs:
        return _PROVIDERS["gpt"]

    for prefix, provider in _PROVIDERS.items():
        if model_name.lower().startswith(prefix):
            return provider

    # Default to OpenAI if no other provider matches
    warnings.warn(
        f"No provider found for model '{model_name}'. Defaulting to OpenAI. "
        f"Supported prefixes are: {list(_PROVIDERS.keys())}"
    )
    return _PROVIDERS["gpt"]


# Set the default model from an environment variable, with a fallback.
DEFAULT_MODEL = os.getenv("ZENLLM_DEFAULT_MODEL", "gpt-4.1")


# ---- Public helpers for building content parts ----

def text(value: Any) -> Dict[str, Any]:
    """Create a text content part."""
    return {"type": "text", "text": str(value)}


def image(source: Any, mime: Optional[str] = None, detail: Optional[str] = None) -> Dict[str, Any]:
    """
    Create an image content part from various sources:
    - str path (e.g., 'photo.jpg') or pathlib.Path
    - str URL (http/https)
    - bytes or bytearray
    - file-like object with .read()
    """
    kind = None
    val = source

    # file-like
    if hasattr(source, "read"):
        kind = "file"
    else:
        # bytes-like
        if isinstance(source, (bytes, bytearray)):
            kind = "bytes"
        else:
            # string or path-like
            if isinstance(source, os.PathLike):
                val = os.fspath(source)
            if isinstance(val, str):
                low = val.lower()
                if low.startswith("http://") or low.startswith("https://"):
                    kind = "url"
                else:
                    kind = "path"
            else:
                raise ValueError("Unsupported image source type. Use a path, URL, bytes, or file-like object.")

    part: Dict[str, Any] = {
        "type": "image",
        "source": {"kind": kind, "value": val},
    }
    if mime:
        part["mime"] = mime
    if detail:
        part["detail"] = detail
    return part


def _normalize_to_parts(content: Union[str, List[Any], None]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if content is None:
        return parts
    if isinstance(content, str):
        parts.append(text(content))
        return parts
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(text(item))
            elif isinstance(item, dict) and "type" in item:
                parts.append(item)
            else:
                raise ValueError("Unsupported content item. Use strings or parts created via text() or image().")
        return parts
    raise ValueError("Unsupported content type. Use a string or list of parts.")


def _normalize_messages(
    prompt_text: Optional[str],
    content: Union[str, List[Any], None],
    messages: Optional[List[Dict[str, Any]]],
    images: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize input into a messages list where each item is:
    {"role": "...", "content": [parts...]}
    """
    if messages:
        normalized: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            m_content = m.get("content")
            parts = _normalize_to_parts(m_content)
            normalized.append({"role": role, "content": parts})
        return normalized

    # Single user-turn composed from prompt_text/content/images
    parts = []
    # Primary text/content
    parts.extend(_normalize_to_parts(content if content is not None else prompt_text))

    # Optional images convenience
    if images:
        for src in images:
            if isinstance(src, dict) and src.get("type") == "image":
                parts.append(src)
            else:
                parts.append(image(src))

    if not parts:
        raise ValueError("You must provide at least prompt_text, content, or messages.")

    return [{"role": "user", "content": parts}]


def prompt(
    prompt_text: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    *,
    content: Union[str, List[Any], None] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    images: Optional[List[Any]] = None,
    **kwargs
):
    """
    A unified function to prompt various LLMs. Backwards compatible with text-only calls,
    and supports multimodal content via text() and image() helpers.

    Args:
        prompt_text (str | None): The user's prompt as plain text.
        model (str): The model to use. Provider inferred from prefix unless base_url is given.
        system_prompt (str | None): System instruction for the model.
        stream (bool): Whether to stream the response.
        content (str | list | None): A string or list of parts (e.g., [text(...), image(...)]). Overrides prompt_text if provided.
        messages (list | None): Full control over the conversation: list of {"role": "...", "content": [parts]}.
        images (list | None): Convenience list of image sources to attach to a single user turn.
        **kwargs: Additional provider-specific parameters.

    Returns:
        str or generator: The model's response as a string, or a generator of strings if streaming.
    """
    provider = _get_provider(model, **kwargs)

    normalized_messages = _normalize_messages(
        prompt_text=prompt_text,
        content=content,
        messages=messages,
        images=images,
    )

    return provider.call(
        model=model,
        messages=normalized_messages,
        system_prompt=system_prompt,
        stream=stream,
        **kwargs
    )