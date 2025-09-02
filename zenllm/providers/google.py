import os
import json
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Tuple

import requests
from .base import LLMProvider

class GoogleProvider(LLMProvider):
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key={api_key}"
    API_KEY_NAME = "GEMINI_API_KEY"
    DEFAULT_MODEL = "gemini-2.5-pro"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your Google API key."
            )
        return api_key

    def _stream_response(self, response):
        def _stream_generator():
            for line in response.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    try:
                        json_str = decoded_line[len('data: '):]
                        data = json.loads(json_str)
                        yield data['candidates'][0]['content']['parts'][0]['text']
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        return _stream_generator()

    # ---- helpers to transform normalized parts to Gemini schema ----

    def _read_image_to_base64(self, part: Dict[str, Any]) -> Tuple[str, str]:
        """
        Return (mime, base64_str) for an image part. If it's a URL, we fetch and encode it.
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
        elif kind == "url":
            # Fetch the image and inline as base64 for Gemini
            resp = requests.get(value, timeout=30)
            resp.raise_for_status()
            data = resp.content
            # Try to infer mime from headers or URL
            mime = mime or resp.headers.get("Content-Type") or mimetypes.guess_type(value)[0] or "image/jpeg"
        else:
            raise ValueError("Unsupported image source for Gemini.")

        if not mime:
            mime = "image/jpeg"

        b64 = base64.b64encode(data).decode("utf-8")
        return mime, b64

    def _to_gemini_parts(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        for p in content:
            if p.get("type") == "text":
                parts.append({"text": p.get("text", "")})
            elif p.get("type") == "image":
                mime, b64 = self._read_image_to_base64(p)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            else:
                # Ignore unknown parts
                continue
        return parts

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()

        contents: List[Dict[str, Any]] = []

        # Build contents with proper roles and parts
        for msg in messages:
            role = msg.get("role", "user")
            parts = msg.get("content")
            if isinstance(parts, list):
                gemini_parts = self._to_gemini_parts(parts)
            else:
                gemini_parts = [{"text": str(parts)}]
            contents.append({
                "role": "user" if role == "user" else "model",
                "parts": gemini_parts
            })

        payload: Dict[str, Any] = {"contents": contents}

        # system_instruction is the official way to set system prompt in Gemini
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        # generationConfig options
        generation_config = {}
        for key in ["temperature", "topP", "topK", "maxOutputTokens"]:
            if key in kwargs:
                generation_config[key] = kwargs.pop(key)
        if generation_config:
            payload["generationConfig"] = generation_config

        payload.update(kwargs)

        method = "streamGenerateContent" if stream else "generateContent"
        api_url = self.API_URL_TEMPLATE.format(
            model=(model or self.DEFAULT_MODEL),
            method=method,
            api_key=api_key
        )

        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)

        response_data = response.json()
        return response_data['candidates'][0]['content']['parts'][0]['text']