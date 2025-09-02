import os
import json
import base64
import mimetypes
from typing import Any, Dict, List, Tuple, Optional

import requests
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    API_URL = "https://api.anthropic.com/v1/messages"
    API_KEY_NAME = "ANTHROPIC_API_KEY"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your Anthropic API key."
            )
        return api_key

    def _stream_response(self, response):
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    try:
                        data = json.loads(decoded_line[len('data: '):])
                        if data.get('type') == 'content_block_delta':
                            yield data['delta']['text']
                    except (json.JSONDecodeError, KeyError):
                        continue

    # ---- helpers to transform normalized parts to Anthropic schema ----

    def _read_image_to_base64(self, part: Dict[str, Any]) -> Tuple[str, str]:
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
        elif kind == "url":
            # Anthropic supports URL sources, but we'll prefer base64 to be consistent
            resp = requests.get(value, timeout=30)
            resp.raise_for_status()
            data = resp.content
            mime = mime or resp.headers.get("Content-Type") or mimetypes.guess_type(value)[0] or "image/jpeg"
        else:
            raise ValueError("Unsupported image source for Anthropic.")

        if not mime:
            mime = "image/jpeg"

        b64 = base64.b64encode(data).decode("utf-8")
        return mime, b64

    def _to_anthropic_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                blocks: List[Dict[str, Any]] = []
                for p in content:
                    if p.get("type") == "text":
                        blocks.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "image":
                        # Use base64 source (works for both path/bytes/file/url)
                        mime, b64 = self._read_image_to_base64(p)
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64
                            }
                        })
                    else:
                        continue
                out.append({"role": role, "content": blocks})
            else:
                # Backwards: plain string content becomes a single text block
                out.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return out

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": model or self.DEFAULT_MODEL,
            "messages": self._to_anthropic_messages(messages),
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": stream,
        }

        if system_prompt:
            payload["system"] = system_prompt

        payload.update(kwargs)

        response = requests.post(self.API_URL, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)

        response_data = response.json()
        return response_data['content'][0]['text']