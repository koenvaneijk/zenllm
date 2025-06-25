import os
import json
import requests
from .base import LLMProvider

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
        """Handles streaming responses from an OpenAI-compatible API."""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[6:].strip()
                    if json_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(json_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content')
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError):
                        continue

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
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)
        
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
        
        response_data = response.json()
        return response_data['choices'][0]['message']['content']