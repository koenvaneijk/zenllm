import os
import requests
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    API_URL = "https://api.openai.com/v1/responses"
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
        """
        Handles streaming responses. The user-provided API docs do not
        specify a streaming format.
        """
        raise NotImplementedError("Streaming is not supported for this OpenAI provider yet.")

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        api_key = self._check_api_key()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

        if not messages:
            raise ValueError("Messages list cannot be empty.")
        
        # This API uses a single 'input' string instead of a messages array.
        # We'll use the content from the last message.
        user_input = messages[-1]['content']

        payload = {
            "model": model or self.DEFAULT_MODEL,
            "input": user_input,
            "stream": stream,
        }
        
        if system_prompt:
            payload["instructions"] = system_prompt
        
        payload.update(kwargs)

        response = requests.post(self.API_URL, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        
        response_data = response.json()
        return response_data['output'][0]['content'][0]['text']