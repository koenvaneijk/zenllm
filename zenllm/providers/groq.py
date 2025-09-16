import os
import json
import base64
import mimetypes
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from .base import LLMProvider, search_pricing_data


GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"


GROQ_PRICING = [
    {
        "model_id": "llama-4-maverick",
        "input_price_per_million_tokens": 0.20,
        "output_price_per_million_tokens": 0.60
    },
    {
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "input_price_per_million_tokens": 1.00,
        "output_price_per_million_tokens": 3.00
    },
    {
        "model_id": "llama-3-8b-8k",
        "input_price_per_million_tokens": 0.05,
        "output_price_per_million_tokens": 0.08
    }
]


class GroqProvider(LLMProvider):
    """
    OpenAI-compatible provider for Groq's Chat Completions API.
    Implements ZenLLM's provider interface with a call(...) method that can stream.
    """

    def _check_api_key(self) -> str:
        key = os.getenv(GROQ_API_KEY_ENV)
        if not key:
            raise ValueError(
                f"Missing Groq API key. Provide api_key=... or set {GROQ_API_KEY_ENV}."
            )
        return key

    def _read_image_to_base64(self, part: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (mime, base64_str) for an 'image' part with non-URL sources.
        """
        src = (part or {}).get("source", {})
        kind = src.get("kind")
        val = src.get("value")
        # file-like
        if kind == "file" and hasattr(val, "read"):
            data = val.read()
            mime = part.get("mime") or "image/png"
            b64 = base64.b64encode(data).decode("utf-8")
            return mime, b64
        # bytes
        if kind == "bytes":
            data = val or b""
            mime = part.get("mime") or "image/png"
            b64 = base64.b64encode(data).decode("utf-8")
            return mime, b64
        # path
        if kind == "path":
            path = str(val)
            mime = part.get("mime") or mimetypes.guess_type(path)[0] or "application/octet-stream"
            with open(path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            return mime, b64
        return None, None

    def _to_openai_messages(self, messages: List[Dict[str, Any]], system_prompt: Optional[str]) -> List[Dict[str, Any]]:
        """
        Convert ZenLLM internal message parts to OpenAI-compatible chat messages.
        Uses content list for multimodal, or plain string for text-only.
        """
        out: List[Dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})

        for m in messages:
            role = m.get("role", "user")
            parts = m.get("content") or []

            # Build content blocks
            content_blocks: List[Dict[str, Any]] = []
            has_image = any(p.get("type") == "image" for p in parts)
            text_acc: List[str] = []

            for p in parts:
                ptype = p.get("type")
                if ptype == "text":
                    t = p.get("text", "")
                    if has_image:
                        if t:
                            content_blocks.append({"type": "text", "text": t})
                    else:
                        text_acc.append(t)
                elif ptype == "image":
                    src = p.get("source", {})
                    if src.get("kind") == "url":
                        url = src.get("value")
                        if url:
                            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        mime, b64 = self._read_image_to_base64(p)
                        if b64:
                            url = f"data:{mime or 'application/octet-stream'};base64,{b64}"
                            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    # Unknown part -> coerce to text
                    s = str(p)
                    if has_image:
                        content_blocks.append({"type": "text", "text": s})
                    else:
                        text_acc.append(s)

            if has_image:
                content = content_blocks
            else:
                # Only text; use simple string for best compatibility
                content = "".join(text_acc)

            # Only allow standard roles
            if role not in ("system", "user", "assistant"):
                role = "user"

            out.append({"role": role, "content": content})
        return out

    def _stream_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """
        Handle SSE streaming from Groq (OpenAI-compatible).
        Yields dict events like {"type":"text","text": "..."}.
        """
        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    line = raw_line.decode("utf-8")
                except Exception:
                    continue
                s = line.strip()
                if not s:
                    continue
                if s.startswith("data:"):
                    s = s[len("data:"):].strip()
                if s == "[DONE]":
                    break
                try:
                    payload = json.loads(s)
                except Exception:
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield {"type": "text", "text": content}
        finally:
            try:
                response.close()
            except Exception:
                pass

    def get_model_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        return search_pricing_data(GROQ_PRICING, model_id)

    def call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ):
        """
        Make a call to Groq's Chat Completions API.
        Returns:
          - if stream=False: dict with keys parts, finish_reason, usage, raw
          - if stream=True: an iterator yielding event dicts
        """
        # Normalize model (allow optional "groq-" prefix)
        normalized_model = model[5:] if model and model.lower().startswith("groq-") else model

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            api_key = self._check_api_key()
        base_url = kwargs.pop("base_url", None) or GROQ_DEFAULT_BASE_URL
        url = f"{base_url.rstrip('/')}/chat/completions"

        # Translate messages
        payload_messages = self._to_openai_messages(messages, system_prompt)

        # Common OpenAI-style options passthrough
        allowed_opts = {
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "logit_bias",
            "user",
            "seed",
        }
        payload: Dict[str, Any] = {"model": normalized_model, "messages": payload_messages}
        for k, v in list(kwargs.items()):
            if k in allowed_opts:
                payload[k] = v

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if stream:
            payload["stream"] = True
            resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=180)
            resp.raise_for_status()
            return self._stream_response(resp)

        # Non-streaming
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        parts: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        try:
            choices = data.get("choices") or []
            if choices:
                first = choices[0]
                msg = first.get("message") or {}
                txt = msg.get("content")
                finish_reason = first.get("finish_reason")
                if txt:
                    parts.append({"type": "text", "text": txt})
            usage = data.get("usage")
        except Exception:
            # Leave parts empty; attach raw for troubleshooting
            pass

        return {
            "parts": parts,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw": data,
        }