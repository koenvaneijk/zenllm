import os
import json
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

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": model or self.DEFAULT_MODEL,
            "messages": messages,
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