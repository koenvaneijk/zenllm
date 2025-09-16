import os
import json
import base64
import mimetypes
from typing import Dict, Any, List, Tuple, Optional

import requests
from .base import LLMProvider, search_pricing_data


OPENAI_PRICING = [
    {
        "model_id": "gpt-5",
        "input_price_per_million_tokens": 1.25,
        "output_price_per_million_tokens": 10.00
    },
    {
        "model_id": "gpt-5-mini",
        "input_price_per_million_tokens": 0.25,
        "output_price_per_million_tokens": 2.00
    },
    {
        "model_id": "gpt-5-nano",
        "input_price_per_million_tokens": 0.05,
        "output_price_per_million_tokens": 0.40
    }
]


class OpenAIProvider(LLMProvider):
    BASE_URL = "https://api.openai.com/v1"
    API_KEY_NAME = "OPENAI_API_KEY"
    DEFAULT_MODEL = "gpt-4.1"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your OpenAI API key."
            )
        return api_key

    def _stream_response(self, response):
        """Handles streaming responses from an OpenAI-compatible API. Yields dict events."""
        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data: '):
                json_str = decoded_line[6:].strip()
                if json_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(json_str)
                    choice = chunk.get('choices', [{}])[0]
                    delta = choice.get('delta', {})
                    content = delta.get('content')
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
            # Guess mime from extension if not provided
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
            if isinstance(content, list):
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
                        # Fallback: ignore unknown parts
                        continue
                out.append({"role": role, "content": parts})
            else:
                # Backwards compatibility: plain string content
                out.append({"role": role, "content": content})
        return out

    def get_model_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        return search_pricing_data(OPENAI_PRICING, model_id)

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        # Pop custom arguments to avoid sending them in the payload
        base_url = kwargs.pop("base_url", self.BASE_URL)
        api_key_override = kwargs.pop("api_key", None)

        full_url = base_url.rstrip('/') + "/chat/completions"

        api_key = api_key_override
        if not api_key:
            # If a custom base_url is used, the API key is optional.
            # If the default base_url is used, the API key from env is required.
            if base_url == self.BASE_URL:
                api_key = self._check_api_key()
            else:
                api_key = os.getenv(self.API_KEY_NAME)

        headers = {
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Construct the final messages payload for Chat Completions
        final_messages: List[Dict[str, Any]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        # Convert normalized parts into OpenAI message schema
        final_messages.extend(self._to_openai_messages(messages))

        if not final_messages:
            raise ValueError("Messages list cannot be empty.")

        payload = {
            "model": model,
            "messages": final_messages,
            "stream": stream,
        }

        payload.update(kwargs)

        response = requests.post(full_url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)

        data = response.json()
        text = data.get('choices', [{}])[0].get('message', {}).get('content', "") or ""
        finish_reason = data.get('choices', [{}])[0].get('finish_reason')
        parts = [{"type": "text", "text": text}] if text else []
        return {
            "parts": parts,
            "raw": data,
            "finish_reason": finish_reason,
            "usage": data.get("usage"),
        }