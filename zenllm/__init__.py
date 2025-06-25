import os
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.deepseek import DeepseekProvider
from .providers.together import TogetherProvider
import warnings

# A mapping from model name prefixes to provider instances
_PROVIDERS = {
    "claude": AnthropicProvider(),
    "gemini": GoogleProvider(),
    "gpt": OpenAIProvider(),
    "deepseek": DeepseekProvider(),
    "together": TogetherProvider(),
}

def _get_provider(model_name, **kwargs):
    """Selects the provider based on the model name or kwargs."""
    # If base_url is provided, always use the OpenAI provider for compatibility.
    if "base_url" in kwargs:
        return _PROVIDERS["gpt"]

    for prefix, provider in _PROVIDERS.items():
        if model_name.lower().startswith(prefix):
            return provider
    
    # Default to OpenAI if no other provider matches
    warnings.warn(f"No provider found for model '{model_name}'. Defaulting to OpenAI. "
                  f"Supported prefixes are: {list(_PROVIDERS.keys())}")
    return _PROVIDERS["gpt"]


# Set the default model from an environment variable, with a fallback.
DEFAULT_MODEL = os.getenv("ZENLLM_DEFAULT_MODEL", "gpt-4.1")


def prompt(
    prompt_text,
    model=DEFAULT_MODEL,
    system_prompt=None,
    stream=False,
    **kwargs
):
    """
    A unified function to prompt various LLMs.

    Args:
        prompt_text (str): The user's prompt.
        model (str, optional): The model to use.
            Defaults to the value of the ZENLLM_DEFAULT_MODEL environment
            variable, or "gpt-4.1" if not set.
            Model name prefix (e.g., 'claude', 'gemini') determines the provider.
        system_prompt (str, optional): An optional system prompt.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        **kwargs: Additional provider-specific parameters.

    Returns:
        str or generator: The model's response as a string, or a generator of strings if streaming.
    """
    provider = _get_provider(model, **kwargs)

    # Standardize the input message format for the provider
    messages = [{"role": "user", "content": prompt_text}]

    return provider.call(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        stream=stream,
        **kwargs
    )