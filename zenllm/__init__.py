import os
import warnings
import base64
from typing import Any, Dict, List, Optional, Union, Iterable, Iterator

from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.deepseek import DeepseekProvider
from .providers.together import TogetherProvider
from .providers.xai import XaiProvider

# ---- Providers registry and selection ----

_PROVIDERS = {
    "claude": AnthropicProvider(),
    "gemini": GoogleProvider(),
    "gpt": OpenAIProvider(),
    "deepseek": DeepseekProvider(),
    "together": TogetherProvider(),
    "xai": XaiProvider(),
    "grok": XaiProvider(),
}

def _get_provider(model_name: Optional[str], provider: Optional[str] = None, **kwargs):
    """
    Select provider by:
      1) explicit provider argument,
      2) presence of base_url (OpenAI-compatible),
      3) model prefix,
      4) default to OpenAI with warning.
    """
    if provider:
        key = provider.lower()
        # Support explicit "openai-compatible"
        if key in ("openai", "gpt", "openai-compatible"):
            return _PROVIDERS["gpt"]
        if key in _PROVIDERS:
            return _PROVIDERS[key]
        warnings.warn(f"Unknown provider '{provider}', defaulting to OpenAI.")
        return _PROVIDERS["gpt"]

    if "base_url" in kwargs:
        return _PROVIDERS["gpt"]

    if model_name:
        for prefix, prov in _PROVIDERS.items():
            if model_name.lower().startswith(prefix):
                return prov

    warnings.warn(
        f"No provider found for model '{model_name}'. Defaulting to OpenAI. "
        f"Supported prefixes are: {list(_PROVIDERS.keys())}"
    )
    return _PROVIDERS["gpt"]


# Default model (can be overridden by env)
DEFAULT_MODEL = os.getenv("ZENLLM_DEFAULT_MODEL", "gpt-4.1")

# ---- Public helpers (escape hatch for advanced parts) ----

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

# ---- Response types ----

class Response:
    def __init__(
        self,
        parts: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        raw: Optional[Dict[str, Any]] = None,
    ):
        self.parts = parts or []
        self.model = model
        self.provider = provider
        self.finish_reason = finish_reason
        self.usage = usage
        self.raw = raw

    @property
    def text(self) -> str:
        return "".join(p.get("text", "") for p in self.parts if p.get("type") == "text")

    @property
    def images(self) -> List[Dict[str, Any]]:
        return [p for p in self.parts if p.get("type") == "image"]

    def save_images(self, dir: str = ".", prefix: str = "img_") -> List[str]:
        os.makedirs(dir, exist_ok=True)
        paths: List[str] = []
        idx = 0
        for p in self.images:
            src = p.get("source", {})
            if src.get("kind") == "bytes":
                data: bytes = src.get("value") or b""
                mime = p.get("mime") or "image/png"
                ext = ".png" if "png" in mime else (".jpg" if "jpeg" in mime or "jpg" in mime else ".bin")
                path = os.path.join(dir, f"{prefix}{idx}{ext}")
                with open(path, "wb") as f:
                    f.write(data)
                paths.append(path)
                idx += 1
        return paths

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe representation: bytes become base64 strings.
        """
        def encode_part(part: Dict[str, Any]) -> Dict[str, Any]:
            if part.get("type") == "image":
                src = part.get("source", {})
                if src.get("kind") == "bytes":
                    b = src.get("value") or b""
                    enc = base64.b64encode(b).decode("utf-8")
                    new = dict(part)
                    new["source"] = {"kind": "bytes_b64", "value": enc}
                    return new
            return part
        return {
            "parts": [encode_part(p) for p in self.parts],
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "raw": self.raw,
        }

class TextEvent:
    type = "text"
    def __init__(self, text: str):
        self.text = text

class ImageEvent:
    type = "image"
    def __init__(self, bytes_data: bytes, mime: Optional[str] = None, url: Optional[str] = None):
        self.bytes = bytes_data
        self.mime = mime or "image/png"
        self.url = url

class ResponseStream:
    def __init__(self, iterator: Iterable, *, model: Optional[str] = None, provider: Optional[str] = None):
        self._it = iter(iterator)
        self._model = model
        self._provider = provider
        self._buffer_text: List[str] = []
        self._image_parts: List[Dict[str, Any]] = []

    def __iter__(self) -> Iterator[Union[TextEvent, ImageEvent]]:
        return self

    def __next__(self) -> Union[TextEvent, ImageEvent]:
        evt = next(self._it)  # may raise StopIteration
        # Normalize dict events into typed objects and accumulate for finalize()
        if isinstance(evt, dict):
            etype = evt.get("type")
            if etype == "text":
                txt = evt.get("text", "")
                self._buffer_text.append(txt)
                return TextEvent(txt)
            if etype == "image":
                if "bytes" in evt:
                    b = evt.get("bytes") or b""
                    mime = evt.get("mime") or "image/png"
                    self._image_parts.append({"type": "image", "source": {"kind": "bytes", "value": b}, "mime": mime})
                    return ImageEvent(b, mime=mime)
                if "url" in evt:
                    url = evt.get("url")
                    mime = evt.get("mime")
                    self._image_parts.append({"type": "image", "source": {"kind": "url", "value": url}, "mime": mime})
                    return ImageEvent(b"", mime=mime, url=url)
        # Text fallback
        if isinstance(evt, str):
            self._buffer_text.append(evt)
            return TextEvent(evt)
        # Unknown event; convert to string
        s = str(evt)
        self._buffer_text.append(s)
        return TextEvent(s)

    def finalize(self) -> Response:
        parts: List[Dict[str, Any]] = []
        text_joined = "".join(self._buffer_text)
        if text_joined:
            parts.append({"type": "text", "text": text_joined})
        parts.extend(self._image_parts)
        return Response(parts, model=self._model, provider=self._provider)

# ---- Input normalization helpers ----

def _normalize_image_source(src: Any) -> Dict[str, Any]:
    if isinstance(src, dict) and src.get("type") == "image":
        return src
    return image(src)

def _message_from_simple(role: str, text_value: Optional[str], images: Optional[Union[Any, List[Any]]]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    if text_value is not None:
        parts.append(text(text_value))
    if images is not None:
        if isinstance(images, list):
            for s in images:
                parts.append(_normalize_image_source(s))
        else:
            parts.append(_normalize_image_source(images))
    return {"role": role, "content": parts}

def _normalize_messages_for_chat(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        # str -> user text
        if isinstance(m, str):
            out.append(_message_from_simple("user", m, None))
            continue
        # tuple -> (role, text, [images?])
        if isinstance(m, tuple):
            if len(m) == 2:
                role, txt = m
                out.append(_message_from_simple(str(role), str(txt), None))
            elif len(m) == 3:
                role, txt, imgs = m
                out.append(_message_from_simple(str(role), str(txt) if txt is not None else None, imgs))
            else:
                raise ValueError("Tuple messages must be (role, text) or (role, text, images).")
            continue
        # dict -> {role,text,images} or {role,parts}
        if isinstance(m, dict):
            role = m.get("role", "user")
            if "parts" in m:
                parts = m.get("parts") or []
                out.append({"role": role, "content": parts})
            else:
                txt = m.get("text")
                imgs = m.get("images")
                out.append(_message_from_simple(role, txt, imgs))
            continue
        raise ValueError("Unsupported message format.")
    return out

# ---- Public API ----

def generate(
    prompt: Optional[str] = None,
    *,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    image: Optional[Any] = None,
    images: Optional[List[Any]] = None,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Single-turn generation with ergonomic inputs. Always returns a Response or ResponseStream.
    - prompt: str text
    - image: single image source (path/URL/bytes/file-like)
    - images: list of image sources
    - options: dict of tuning and passthrough
    """
    # Build a single user message
    msg = _message_from_simple("user", prompt, images if images is not None else image)
    msgs = [msg]

    # Prepare kwargs/options passthrough
    kwargs: Dict[str, Any] = {}
    if options:
        kwargs.update(options)
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    prov = _get_provider(model, provider=provider, **kwargs)
    prov_name = prov.__class__.__name__.replace("Provider", "").lower()

    if stream:
        iterator = prov.call(model=model, messages=msgs, system_prompt=system, stream=True, **kwargs)
        return ResponseStream(iterator, model=model, provider=prov_name)

    # Non-stream
    result = prov.call(model=model, messages=msgs, system_prompt=system, stream=False, **kwargs)
    parts = result.get("parts") or []
    return Response(
        parts,
        model=model,
        provider=prov_name,
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage"),
        raw=result.get("raw"),
    )

def chat(
    messages: List[Any],
    *,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    stream: bool = False,
    options: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Multi-turn chat with ergonomic shorthands.
    messages accepts:
      - "hello"
      - ("user"|"assistant"|"system", text[, images])
      - {"role":"user","text":"...", "images":[...]}
      - {"role":"user","parts":[...]}  # escape hatch
    """
    msgs = _normalize_messages_for_chat(messages)

    kwargs: Dict[str, Any] = {}
    if options:
        kwargs.update(options)
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    prov = _get_provider(model, provider=provider, **kwargs)
    prov_name = prov.__class__.__name__.replace("Provider", "").lower()

    if stream:
        iterator = prov.call(model=model, messages=msgs, system_prompt=system, stream=True, **kwargs)
        return ResponseStream(iterator, model=model, provider=prov_name)

    result = prov.call(model=model, messages=msgs, system_prompt=system, stream=False, **kwargs)
    parts = result.get("parts") or []
    return Response(
        parts,
        model=model,
        provider=prov_name,
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage"),
        raw=result.get("raw"),
    )