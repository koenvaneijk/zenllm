import os
import json
import base64
import mimetypes
from typing import Dict, Any, List, Tuple, Optional

import requests
from .base import LLMProvider


class XaiProvider(LLMProvider):
    BASE_URL = "https://api.x.ai/v1"
    API_KEY_NAME = "XAI_API_KEY"
    DEFAULT_MODEL = "grok-4-fast"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your X.ai API key."
            )
        return api_key

    def _stream_response(self, response):
        """Handles streaming responses from X.ai (OpenAI-compatible). Yields dict events."""
        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode("utf-8", errors="ignore") if isinstance(line, (bytes, bytearray)) else str(line)
            if decoded_line.startswith("data: "):
                json_str = decoded_line[6:].strip()
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        yield {"type": "text", "text": content}
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

    # ---- helpers to transform normalized parts to OpenAI schema ----

    def _read_image_to_base64(self, part: Dict[str, Any]) -> Tuple[str, str]:
        """
        Return (mime, base64_str) for an image part with non-URL sources.
        """
        source = part.get("source", {})
        kind = source.get("kind")
        value = source.get("value")
        mime = part.get("mime")

        data: Optional[bytes] = None
        if kind == "bytes":
            data = value if isinstance(value, (bytes, bytearray)) else bytes(value)
        elif kind == "file":
            data = value.read()
        elif kind == "path":
            if not mime and isinstance(value, str):
                mime = mimetypes.guess_type(value)[0] or "image/jpeg"
            with open(value, "rb") as f:
                data = f.read()
        else:
            raise ValueError("Unsupported image source for base64 conversion.")

        if not mime:
            mime = "image/jpeg"

        b64 = base64.b64encode(data).decode("utf-8")
        return mime, b64

    def _to_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if not isinstance(content, list):
                # Always convert to list format for X.ai compatibility
                content = [{"type": "text", "text": content or ""}]
            parts: List[Dict[str, Any]] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") == "image":
                    source = p.get("source", {})
                    kind = source.get("kind")
                    detail = p.get("detail")
                    if kind == "url":
                        url = source.get("value")
                        image_url_obj: Dict[str, Any] = {"url": url}
                        if detail:
                            image_url_obj["detail"] = detail
                        parts.append({"type": "image_url", "image_url": image_url_obj})
                    else:
                        mime, b64 = self._read_image_to_base64(p)
                        data_url = f"data:{mime};base64,{b64}"
                        image_url_obj = {"url": data_url}
                        if detail:
                            image_url_obj["detail"] = detail
                        parts.append({"type": "image_url", "image_url": image_url_obj})
                else:
                    # Ignore unknown parts
                    continue
            out.append({"role": role, "content": parts})
        return out

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        # Allow overrides similar to OpenAI provider
        base_url = kwargs.pop("base_url", self.BASE_URL)
        api_key_override = kwargs.pop("api_key", None)

        full_url = base_url.rstrip("/") + "/chat/completions"

        api_key = api_key_override or os.getenv(self.API_KEY_NAME)
        if base_url == self.BASE_URL and not api_key:
            api_key = self._check_api_key()

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Construct the final messages payload for Chat Completions
        final_messages: List[Dict[str, Any]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(self._to_openai_messages(messages))

        if not final_messages:
            raise ValueError("Messages list cannot be empty.")

        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        payload = {
            "model": model or self.DEFAULT_MODEL,
            "messages": final_messages,
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        payload.update(kwargs)

        response = requests.post(full_url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            msg = choice.get("message", {}) or {}
            text = msg.get("content") or ''
            tool_calls = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            parts = [{"type": "text", "text": text}] if text else []
            return {
                "parts": parts,
                "tool_calls": tool_calls,
                "raw": data,
                "finish_reason": finish_reason,
                "usage": data.get("usage"),
            }
        return {
            "parts": [],
            "tool_calls": None,
            "raw": data,
            "finish_reason": None,
            "usage": data.get("usage"),
        }