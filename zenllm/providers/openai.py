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
        # Pop custom arguments to avoid sending them in the payload
        api_url = kwargs.pop("api_url", self.API_URL)
        api_key_override = kwargs.pop("api_key", None)

        api_key = api_key_override
        if not api_key:
            # If a custom URL is used, the API key is optional (e.g., local models).
            # If the default URL is used, the API key from env is required.
            if api_url == self.API_URL:
                api_key = self._check_api_key()
            else:
                api_key = os.getenv(self.API_KEY_NAME)

        headers = {
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

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

        response = requests.post(api_url, headers=headers, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        
        response_data = response.json()
        return response_data['output'][0]['content'][0]['text']