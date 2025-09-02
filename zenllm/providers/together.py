import os
import json
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Tuple

import requests
from .base import LLMProvider

class TogetherProvider(LLMProvider):
    API_URL = "https://api.together.xyz/v1/chat/completions"
    API_KEY_NAME = "TOGETHER_API_KEY"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your TogetherAI API key."
            )
        return api_key

    def _stream_response(self, response):
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    content = decoded_line[len('data: '):]
                    if content.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(content)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content'] is not None:
                                yield delta['content']
                    except (json.JSONDecodeError, KeyError):
                        continue

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
        else:
            # Avoid fetching external URLs automatically for Together
            raise ValueError("Together provider does not support URL images; use path/bytes/file.")
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
                    if p.get("type") == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") == "image":
                        source = p.get("source", {})
                        if source.get("kind") == "url":
                            raise ValueError("Together provider does not support URL images. Use path/bytes/file.")
                        mime, b64 = self._read_image_to_base64(p)
                        data_url = f"data:{mime};base64,{b64}"
                        parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    else:
                        continue
                out.append({"role": role, "content": parts})
            else:
                out.append({"role": role, "content": content})
        return out

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Remove the 'together/' prefix for the API call
        if model.startswith("together/"):
            api_model = model.split('/', 1)[1]
        else:
            api_model = model

        final_messages: List[Dict[str, Any]] = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(self._to_openai_messages(messages))

        payload = {
            "model": api_model,
            "messages": final_messages,
            "stream": stream,
        }
        
        payload.update(kwargs)

        response = requests.post(self.API_URL, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        
        response_data = response.json()
        return response_data['choices'][0]['message']['content']