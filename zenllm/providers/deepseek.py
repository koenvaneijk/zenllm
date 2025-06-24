import os
import json
import requests
from .base import LLMProvider

class DeepseekProvider(LLMProvider):
    API_URL = "https://api.deepseek.com/chat/completions"
    API_KEY_NAME = "DEEPSEEK_API_KEY"
    DEFAULT_MODEL = "deepseek-chat"

    def _check_api_key(self):
        api_key = os.getenv(self.API_KEY_NAME)
        if not api_key:
            raise ValueError(
                f"{self.API_KEY_NAME} environment variable not set. "
                "Please set it to your DeepSeek API key."
            )
        return api_key

    def _stream_response(self, response):
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[len('data: '):].strip()
                    if json_str == '[DONE]':
                        break
                    if not json_str:
                        continue
                    try:
                        data = json.loads(json_str)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": model or self.DEFAULT_MODEL,
            "messages": all_messages,
            "stream": stream,
        }
        
        payload.update(kwargs)

        response = requests.post(self.API_URL, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        
        response_data = response.json()
        return response_data['choices'][0]['message']['content']