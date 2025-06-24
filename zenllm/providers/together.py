import os
import json
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

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": api_model,
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