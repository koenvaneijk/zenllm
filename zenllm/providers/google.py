import os
import json
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

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()
        
        contents = []
        if system_prompt:
             contents.append({"role": "user", "parts": [{"text": system_prompt}]})
             contents.append({"role": "model", "parts": [{"text": "Understood."}]})

        for msg in messages:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
            
        payload = {"contents": contents}
        
        generation_config = {}
        for key in ["temperature", "topP", "topK", "maxOutputTokens"]:
             if key in kwargs:
                 generation_config[key] = kwargs.pop(key)
        if generation_config:
            payload["generationConfig"] = generation_config
            
        payload.update(kwargs)

        method = "streamGenerateContent" if stream else "generateContent"
        api_url = self.API_URL_TEMPLATE.format(model=model or self.DEFAULT_MODEL, method=method, api_key=api_key)

        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        
        response_data = response.json()
        return response_data['candidates'][0]['content']['parts'][0]['text']