import os
import warnings
import base64
import time
import random
import requests
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterable, Iterator, Callable, Tuple, get_origin, get_args

from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.deepseek import DeepseekProvider
from .providers.together import TogetherProvider
from .providers.xai import XaiProvider
from .providers.groq import GroqProvider, GROQ_DEFAULT_BASE_URL, GROQ_API_KEY_ENV
from .pricing import estimate_cost as _estimate_cost

# ---- Providers registry and selection ----

_PROVIDERS = {
    "claude": AnthropicProvider(),
    "gemini": GoogleProvider(),
    "gpt": OpenAIProvider(),
    "deepseek": DeepseekProvider(),
    "together": TogetherProvider(),
    "xai": XaiProvider(),
    "grok": XaiProvider(),
    "groq": GroqProvider(),
}

_PROVIDER_BY_NAME = {
    prov.__class__.__name__.replace("Provider", "").lower(): prov
    for prov in set(_PROVIDERS.values())
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

# ---- Models listing API ----

@dataclass
class ModelInfo:
    id: str
    created: Optional[int] = None
    owned_by: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

def list_models(
    provider: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[ModelInfo]:
    """
    List models available for a provider.

    Implemented providers:
      - OpenAI-compatible endpoints (provider: None | "openai" | "gpt" | "openai-compatible", or if base_url is provided)
      - Groq (provider: "groq")
      - Anthropic (provider: "anthropic" | "claude")
      - DeepSeek (provider: "deepseek")
      - Google Gemini (provider: "gemini")
      - Together (provider: "together")
      - X.ai (provider: "xai" | "grok")

    Arguments:
      - provider: which provider to query
      - base_url: override base URL (for the given provider when supported)
      - api_key: explicit API key; defaults to provider-specific env var

    Returns:
      - List[ModelInfo]
    """
    prov_key = (provider or "").lower() if provider else None

    # OpenAI-compatible path (default if base_url is provided or provider is None/openai/gpt/openai-compatible)
    if (base_url is not None) or (prov_key in (None, "openai", "gpt", "openai-compatible")):
        url = f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/models"
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing API key for listing models. Provide api_key=... or set OPENAI_API_KEY.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") or []

    # Groq
    elif prov_key == "groq":
        url = f"{(base_url or GROQ_DEFAULT_BASE_URL).rstrip('/')}/models"
        key = api_key or os.getenv(GROQ_API_KEY_ENV)
        if not key:
            raise ValueError(f"Missing API key for Groq model listing. Provide api_key=... or set {GROQ_API_KEY_ENV}.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if items is None:
            items = []

    # Anthropic
    elif prov_key in ("anthropic", "claude"):
        url = f"{(base_url or 'https://api.anthropic.com/v1').rstrip('/')}/models"
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Missing API key for Anthropic model listing. Provide api_key=... or set ANTHROPIC_API_KEY.")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if items is None:
            items = []

    # DeepSeek
    elif prov_key == "deepseek":
        url = f"{(base_url or 'https://api.deepseek.com').rstrip('/')}/models"
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("Missing API key for DeepSeek model listing. Provide api_key=... or set DEEPSEEK_API_KEY.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if items is None:
            items = []

    # Google Gemini (OpenAI-compatible models endpoint)
    elif prov_key in ("gemini", "google"):
        url = f"{(base_url or 'https://generativelanguage.googleapis.com').rstrip('/')}/v1beta/openai/models"
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing API key for Google Gemini model listing. Provide api_key=... or set GEMINI_API_KEY.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if items is None:
            items = []

    # Together AI
    elif prov_key in ("together", "togetherai"):
        url = f"{(base_url or 'https://api.together.xyz/v1').rstrip('/')}/models"
        key = api_key or os.getenv("TOGETHER_API_KEY")
        if not key:
            raise ValueError("Missing API key for Together model listing. Provide api_key=... or set TOGETHER_API_KEY.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        # Together may return a top-level array or an object with data
        if isinstance(payload, list):
            items = payload
        else:
            items = payload.get("data") or []

    # X.ai (also alias "grok")
    elif prov_key in ("xai", "grok"):
        url = f"{(base_url or 'https://api.x.ai/v1').rstrip('/')}/models"
        key = api_key or os.getenv("XAI_API_KEY")
        if not key:
            raise ValueError("Missing API key for X.ai model listing. Provide api_key=... or set XAI_API_KEY.")
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if items is None:
            items = []

    else:
        raise NotImplementedError("list_models is implemented for OpenAI-compatible providers, Groq, Anthropic, DeepSeek, Gemini, Together, and X.ai.")

    models: List[ModelInfo] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = it.get("id")
        if not mid:
            continue
        created_val = it.get("created")
        # Some providers use created_at (ISO string); we keep created=None to avoid extra deps for parsing
        owned_by = it.get("owned_by") or it.get("organization")
        models.append(ModelInfo(
            id=mid,
            created=created_val if isinstance(created_val, int) else None,
            owned_by=owned_by,
            raw=it,
        ))
    return models

# ---- Fallback configuration ----

@dataclass
class ProviderChoice:
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

@dataclass
class RetryPolicy:
    max_attempts: int = 1
    initial_backoff: float = 0.5
    max_backoff: float = 4.0
    timeout: Optional[float] = None  # placeholder; some providers may accept request timeouts via options

@dataclass
class FallbackConfig:
    chain: List[ProviderChoice]
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    allow_mid_stream_switch: bool = False
    # Status codes considered non-retryable (if provided by providers)
    non_retryable_statuses: Optional[List[int]] = None
    # Additional retryable statuses (beyond defaults)
    retryable_statuses: Optional[List[int]] = None

def _env_default_fallback() -> Optional["FallbackConfig"]:
    """
    Parse ZENLLM_FALLBACK="provider:model,provider:model,..." into a FallbackConfig.
    """
    spec = os.getenv("ZENLLM_FALLBACK")
    if not spec:
        return None
    chain: List[ProviderChoice] = []
    for item in spec.split(","):
        tok = item.strip()
        if not tok:
            continue
        prov = None
        model = None
        if ":" in tok:
            prov, model = tok.split(":", 1)
            prov = (prov or None)
            model = (model or None)
        else:
            # If only one token provided, assume it's a model or provider; we'll infer later
            if tok in _PROVIDERS or tok in ("openai", "gpt", "openai-compatible"):
                prov = tok
            else:
                model = tok
        chain.append(ProviderChoice(provider=prov, model=model))
    if not chain:
        return None
    return FallbackConfig(chain=chain)

def _status_from_exception(exc: Exception) -> Optional[int]:
    """
    Try to extract an HTTP-like status code from a provider exception.
    """
    # Common patterns: exc.status_code, exc.status, exc.response.status_code
    for attr in ("status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            v = getattr(resp, attr, None)
            if isinstance(v, int):
                return v
    return None

def _is_retryable(exc: Exception, status: Optional[int], cfg: FallbackConfig) -> bool:
    default_non_retryable = {400, 401, 403, 404, 422}
    default_retryable = {408, 429}
    default_retryable.update({s for s in range(500, 600)})

    if cfg.non_retryable_statuses is not None:
        non_retryable = set(cfg.non_retryable_statuses)
    else:
        non_retryable = default_non_retryable
    retryable = set(default_retryable)
    if cfg.retryable_statuses is not None:
        retryable.update(cfg.retryable_statuses)

    if status is not None:
        if status in non_retryable:
            return False
        if status in retryable:
            return True
        # Unknown status: be conservative and retry on 5xx-like statuses already handled above
        return False

    # No status: consider network/timeouts retryable; fall back to type checks
    # Avoid retrying on clear client-side errors like ValueError/TypeError
    if isinstance(exc, (ValueError, TypeError)):
        return False
    # Otherwise assume retryable (connection reset, timeouts, etc.)
    return True

def _backoff_sleep(attempt_index: int, retry: RetryPolicy):
    # attempt_index is 0-based for backoff computation
    base = min(retry.max_backoff, retry.initial_backoff * (2 ** attempt_index))
    # Full jitter
    time.sleep(random.uniform(0, base))

def _merge_options(global_opts: Optional[Dict[str, Any]], choice_opts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if global_opts:
        merged.update(global_opts)
    if choice_opts:
        merged.update(choice_opts)
    return merged

def _prov_name(prov_obj) -> str:
    return prov_obj.__class__.__name__.replace("Provider", "").lower()

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

# ---- Tools decorator and agent (autorun default false) ----

def _doc_first_line(obj: Any) -> Optional[str]:
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return None
    # First non-empty line
    for line in doc.splitlines():
        s = line.strip()
        if s:
            return s
    return None

def _is_typed_dict(tp: Any) -> bool:
    # Best-effort detection for TypedDict subclasses at runtime
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(tp, "__total__")

def _is_optional(tp: Any) -> bool:
    if tp is None:
        return True
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        return any(a is type(None) for a in args)  # noqa: E721
    return False

def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return args[0]
    return tp

def _type_to_schema(tp: Any) -> Dict[str, Any]:
    # Default fallback
    if tp is inspect._empty or tp is Any or tp is None:
        return {"type": "string"}
    # Handle Optional[X]
    if _is_optional(tp):
        inner = _unwrap_optional(tp)
        sch = _type_to_schema(inner)
        # Do not mark required at the schema level; required handled outside
        return sch

    origin = get_origin(tp)
    if origin in (list, List):
        (item_tp,) = get_args(tp) or (Any,)
        return {"type": "array", "items": _type_to_schema(item_tp)}
    if origin in (dict, Dict):
        args = get_args(tp)
        key_tp = args[0] if args else str
        val_tp = args[1] if args and len(args) > 1 else Any
        # JSON object keys are strings
        if key_tp not in (str, Any):
            # Coerce to string keys; we still describe value type
            pass
        return {"type": "object", "additionalProperties": _type_to_schema(val_tp)}
    if origin in (tuple, Tuple):
        args = get_args(tp) or ()
        items = [_type_to_schema(a) for a in args]
        return {"type": "array", "prefixItems": items, "items": False}

    # Primitives
    if tp in (str,):
        return {"type": "string"}
    if tp in (int,):
        return {"type": "integer"}
    if tp in (float,):
        return {"type": "number"}
    if tp in (bool,):
        return {"type": "boolean"}

    # TypedDict-like
    if _is_typed_dict(tp):
        props: Dict[str, Any] = {}
        req: List[str] = []
        anns = getattr(tp, "__annotations__", {}) or {}
        total = bool(getattr(tp, "__total__", True))
        for k, v in anns.items():
            props[k] = _type_to_schema(v)
            # If total=True -> required unless Optional
            if total and not _is_optional(v):
                req.append(k)
        schema: Dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
        if req:
            schema["required"] = req
        return schema

    # Dataclass-like or objects with __annotations__
    if hasattr(tp, "__annotations__"):
        props: Dict[str, Any] = {}
        req: List[str] = []
        anns = getattr(tp, "__annotations__", {}) or {}
        for k, v in anns.items():
            props[k] = _type_to_schema(v)
            if not _is_optional(v):
                req.append(k)
        schema = {"type": "object", "properties": props}
        if req:
            schema["required"] = req
        return schema

    # Fallback to string
    return {"type": "string"}

def _build_parameters_schema(fn: Callable) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        # Skip *args/**kwargs for schema
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = param.annotation
        props[name] = _type_to_schema(ann)
        # Required if no default and not Optional
        if param.default is inspect._empty and not _is_optional(ann):
            required.append(name)
    schema: Dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema

def _build_tool_spec(
    fn: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    safe: bool = False,
) -> Dict[str, Any]:
    tname = name or fn.__name__
    desc = description or _doc_first_line(fn) or ""
    params_schema = parameters or _build_parameters_schema(fn)
    return {
        "name": tname,
        "description": desc,
        "parameters": params_schema,
        "safe": bool(safe),
        "executor": fn,
    }

def tool(_fn: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None, safe: bool = False):
    """
    Decorator to register a Python function as an LLM-callable tool.

    Example:
      @zenllm.tool
      def get_weather(city: str) -> dict:
          "Get current weather by city."
          ...

    The wrapped function will have .__zen_tool__ with:
      {name, description, parameters (JSON Schema), safe, executor}
    """
    def _wrap(fn: Callable):
        spec = _build_tool_spec(fn, name=name, description=description, parameters=parameters, safe=safe)
        setattr(fn, "__zen_tool__", spec)
        return fn

    if _fn is None:
        return _wrap
    return _wrap(_fn)

def _coerce_to_tool_spec(obj: Any) -> Dict[str, Any]:
    """
    Accepts:
      - a function decorated with @tool (uses __zen_tool__)
      - any callable (derive a spec from signature)
      - a prebuilt dict spec with keys name, description, parameters
    Returns a normalized internal spec with executor where available.
    """
    if isinstance(obj, dict):
        spec = dict(obj)
        # Ensure minimal fields
        if "name" not in spec:
            raise ValueError("Tool dict must include a 'name'.")
        if "parameters" not in spec:
            spec["parameters"] = {"type": "object", "properties": {}, "additionalProperties": True}
        spec.setdefault("description", "")
        # No executor for raw dicts
        return {"executor": None, "safe": False, **spec}
    if callable(obj):
        if hasattr(obj, "__zen_tool__"):
            return getattr(obj, "__zen_tool__")
        # Derive from signature
        return _build_tool_spec(obj)
    raise ValueError("Unsupported tool type; expected callable or dict spec.")

def _to_openai_tool_dict(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description") or "",
            "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
        },
    }

def _prepare_tools(tools: Optional[List[Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Callable], List[Dict[str, Any]]]:
    """
    Returns:
      - specs: internal normalized tool specs (with executor when available)
      - exec_map: name -> executor (only for callable tools)
      - request_tools: OpenAI-compatible tool list for options
    """
    specs: List[Dict[str, Any]] = []
    exec_map: Dict[str, Callable] = {}
    req: List[Dict[str, Any]] = []
    if not tools:
        return specs, exec_map, req
    for t in tools:
        spec = _coerce_to_tool_spec(t)
        specs.append(spec)
        if spec.get("executor"):
            exec_map[spec["name"]] = spec["executor"]
        req.append(_to_openai_tool_dict(spec))
    return specs, exec_map, req

def agent(
    messages: List[Any],
    *,
    tools: Optional[List[Any]] = None,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    fallback: Optional["FallbackConfig"] = None,
    stream: bool = False,
    auto_run_tools: bool = False,  # default false as requested
):
    """
    High-level helper to run a tool-enabled chat.
    - Wraps chat() and passes tool definitions via options.
    - By default does NOT auto-execute tools (auto_run_tools=False).
    - If auto_run_tools=True: currently not implemented end-to-end without provider tool-call parsing.
      We will pass tool definitions and return the first assistant message.

    Returns: Response or ResponseStream from chat().
    """
    # Normalize messages using existing helper
    msgs = _normalize_messages_for_chat(messages)

    # Prepare tools for request
    _, _, request_tools = _prepare_tools(tools)

    # Merge options with tools/tool_choice
    opts: Dict[str, Any] = {}
    if options:
        opts.update(options)
    if request_tools:
        opts["tools"] = request_tools
        # Let the model decide when to call
        # Many providers accept "auto" | "none" | {"type":"function","function":{"name":...}}
        # We default to auto, but since auto_run_tools=False, we won't execute the call client-side.
        if "tool_choice" not in opts:
            opts["tool_choice"] = "auto"

    if auto_run_tools:
        warnings.warn(
            "auto_run_tools=True is not yet fully implemented; executing tools requires provider tool-call support. "
            "Tool definitions are sent to the model, but tool execution/looping is not performed in this version."
        )

    # Delegate to core chat for now
    return chat(
        msgs,
        model=model,
        system=system,
        stream=stream,
        options=opts,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        fallback=fallback,
    )

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
        self._cost_cache: Optional[Dict[str, Any]] = None

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

    def cost_breakdown(self, *, prompt_chars: Optional[int] = None, completion_chars: Optional[int] = None) -> Dict[str, Any]:
        """
        Return a cost breakdown dict for this response. Uses provider-reported usage when available.
        If prompt_chars/completion_chars are provided, they are used to approximate missing token counts.
        """
        # Cache only the no-args call to keep deterministic behavior when char-based approximations are passed
        if prompt_chars is None and completion_chars is None and self._cost_cache is not None:
            return self._cost_cache
        result = _estimate_cost(
            model=self.model,
            usage=self.usage,
            prompt_chars=prompt_chars,
            completion_chars=completion_chars if completion_chars is not None else len(self.text) if self.text else None,
            provider_name=self.provider,
            provider_map=_PROVIDER_BY_NAME,
        )
        if prompt_chars is None and completion_chars is None:
            self._cost_cache = result
        return result

    def cost(self, *, prompt_chars: Optional[int] = None, completion_chars: Optional[int] = None) -> Optional[float]:
        """
        Return the total USD cost for this response (None if pricing unknown).
        """
        breakdown = self.cost_breakdown(prompt_chars=prompt_chars, completion_chars=completion_chars)
        return breakdown.get("total")

    


    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe representation: bytes become base64 strings.
        Includes 'cost' (float) and 'cost_breakdown' computed from usage/pricing when possible.
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
            "cost": self.cost(),  # float total or None
            "cost_breakdown": self.cost_breakdown(),
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
    def __init__(self, iterator: Iterable, *, model: Optional[str] = None, provider: Optional[str] = None, raw: Optional[Dict[str, Any]] = None):
        self._it = iter(iterator)
        self._model = model
        self._provider = provider
        self._raw = raw
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
        return Response(parts, model=self._model, provider=self._provider, raw=self._raw)

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

# ---- Fallback runner ----

def _run_with_fallback(
    *,
    msgs: List[Dict[str, Any]],
    default_model: str,
    system: Optional[str],
    stream: bool,
    options: Optional[Dict[str, Any]],
    fallback: FallbackConfig,
    default_provider: Optional[str],
    default_base_url: Optional[str],
    default_api_key: Optional[str],
):
    attempts_log: List[Dict[str, Any]] = []

    def choice_kwargs(choice: ProviderChoice) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        # Merge options (global < choice)
        kw.update(_merge_options(options, choice.options))
        # Base URL / API key precedence: choice overrides call-level
        if choice.base_url or default_base_url:
            kw["base_url"] = choice.base_url or default_base_url
        if choice.api_key or default_api_key:
            kw["api_key"] = choice.api_key or default_api_key
        return kw

    # Non-stream path
    if not stream:
        for choice in fallback.chain:
            model = choice.model or default_model
            kw = choice_kwargs(choice)
            prov = _get_provider(model, provider=choice.provider or default_provider, **kw)
            prov_name = _prov_name(prov)

            for attempt in range(1, (fallback.retry.max_attempts or 1) + 1):
                try:
                    result = prov.call(model=model, messages=msgs, system_prompt=system, stream=False, **kw)
                    parts = result.get("parts") or []
                    raw_meta = {
                        "fallback": {
                            "selected_provider": prov_name,
                            "selected_model": model,
                            "attempts": attempts_log,
                        },
                        "provider_raw": result.get("raw"),
                    }
                    return Response(
                        parts,
                        model=model,
                        provider=prov_name,
                        finish_reason=result.get("finish_reason"),
                        usage=result.get("usage"),
                        raw=raw_meta,
                    )
                except Exception as e:
                    status = _status_from_exception(e)
                    retryable = _is_retryable(e, status, fallback)
                    attempts_log.append({
                        "provider": prov_name,
                        "model": model,
                        "attempt": attempt,
                        "status": status,
                        "retryable": retryable,
                        "error": str(e),
                    })
                    # Stop retrying this provider if not retryable or out of attempts
                    if not retryable or attempt >= (fallback.retry.max_attempts or 1):
                        break
                    _backoff_sleep(attempt - 1, fallback.retry)
            # move to next provider
        # Exhausted all providers
        raise RuntimeError(f"All providers failed. Attempts: {attempts_log}")

    # Stream path
    if not fallback.allow_mid_stream_switch:
        # Lock in provider after first event arrives
        for choice in fallback.chain:
            model = choice.model or default_model
            kw = choice_kwargs(choice)
            prov = _get_provider(model, provider=choice.provider or default_provider, **kw)
            prov_name = _prov_name(prov)

            for attempt in range(1, (fallback.retry.max_attempts or 1) + 1):
                try:
                    iterator = prov.call(model=model, messages=msgs, system_prompt=system, stream=True, **kw)
                    it = iter(iterator)
                    # Prefetch first event to ensure provider is viable
                    try:
                        first = next(it)
                        # Build generator that replays first then continues
                        def gen():
                            yield first
                            for ev in it:
                                yield ev
                        raw_meta = {
                            "fallback": {
                                "selected_provider": prov_name,
                                "selected_model": model,
                                "attempts": attempts_log,
                            }
                        }
                        return ResponseStream(gen(), model=model, provider=prov_name, raw=raw_meta)
                    except StopIteration:
                        # Empty stream; treat as success with empty content
                        def gen_empty():
                            if False:
                                yield None
                        raw_meta = {
                            "fallback": {
                                "selected_provider": prov_name,
                                "selected_model": model,
                                "attempts": attempts_log,
                            }
                        }
                        return ResponseStream(gen_empty(), model=model, provider=prov_name, raw=raw_meta)
                    except Exception as iter_exc:
                        status = _status_from_exception(iter_exc)
                        retryable = _is_retryable(iter_exc, status, fallback)
                        attempts_log.append({
                            "provider": prov_name,
                            "model": model,
                            "attempt": attempt,
                            "status": status,
                            "retryable": retryable,
                            "error": str(iter_exc),
                        })
                        if not retryable or attempt >= (fallback.retry.max_attempts or 1):
                            break
                        _backoff_sleep(attempt - 1, fallback.retry)
                        continue
                except Exception as e:
                    status = _status_from_exception(e)
                    retryable = _is_retryable(e, status, fallback)
                    attempts_log.append({
                        "provider": prov_name,
                        "model": model,
                        "attempt": attempt,
                        "status": status,
                        "retryable": retryable,
                        "error": str(e),
                    })
                    if not retryable or attempt >= (fallback.retry.max_attempts or 1):
                        break
                    _backoff_sleep(attempt - 1, fallback.retry)
                    continue
        raise RuntimeError(f"All providers failed (stream preflight). Attempts: {attempts_log}")

    # Advanced: allow mid-stream switch (restart on next provider if failure mid-stream)
    # Note: This is simplistic: we do not splice streams; we restart from scratch on next provider.
    def switching_stream():
        for choice in fallback.chain:
            model = choice.model or default_model
            kw = choice_kwargs(choice)
            prov = _get_provider(model, provider=choice.provider or default_provider, **kw)
            prov_name = _prov_name(prov)
            for attempt in range(1, (fallback.retry.max_attempts or 1) + 1):
                try:
                    iterator = prov.call(model=model, messages=msgs, system_prompt=system, stream=True, **kw)
                    for ev in iterator:
                        yield ev
                    # Completed successfully
                    return
                except Exception as e:
                    status = _status_from_exception(e)
                    retryable = _is_retryable(e, status, fallback)
                    attempts_log.append({
                        "provider": prov_name,
                        "model": model,
                        "attempt": attempt,
                        "status": status,
                        "retryable": retryable,
                        "error": str(e),
                    })
                    if not retryable or attempt >= (fallback.retry.max_attempts or 1):
                        break
                    _backoff_sleep(attempt - 1, fallback.retry)
                    continue
        # If exhausted all, end iteration (consumer finalize will see partial)
        return

    raw_meta = {"fallback": {"selected_provider": None, "selected_model": None, "attempts": attempts_log}}
    return ResponseStream(switching_stream(), model=default_model, provider=None, raw=raw_meta)

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
    fallback: Optional[FallbackConfig] = None,
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

    # If fallback provided or env default exists, use fallback runner
    fb = fallback or _env_default_fallback()
    if fb:
        return _run_with_fallback(
            msgs=msgs,
            default_model=model,
            system=system,
            stream=stream,
            options=options,
            fallback=fb,
            default_provider=provider,
            default_base_url=base_url,
            default_api_key=api_key,
        )

    # Prepare kwargs/options passthrough
    kwargs: Dict[str, Any] = {}
    if options:
        kwargs.update(options)
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    prov = _get_provider(model, provider=provider, **kwargs)
    prov_name = _prov_name(prov)

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
    fallback: Optional[FallbackConfig] = None,
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

    fb = fallback or _env_default_fallback()
    if fb:
        return _run_with_fallback(
            msgs=msgs,
            default_model=model,
            system=system,
            stream=stream,
            options=options,
            fallback=fb,
            default_provider=provider,
            default_base_url=base_url,
            default_api_key=api_key,
        )

    kwargs: Dict[str, Any] = {}
    if options:
        kwargs.update(options)
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    prov = _get_provider(model, provider=provider, **kwargs)
    prov_name = _prov_name(prov)

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