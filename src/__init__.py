from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
import warnings

# A mapping from model name prefixes to provider instances
_PROVIDERS = {
    "claude": AnthropicProvider(),
    "gemini": GoogleProvider(),
}

def _get_provider(model_name):
    """Selects the provider based on the model name."""
    for prefix, provider in _PROVIDERS.items():
        if model_name.lower().startswith(prefix):
            return provider
    
    # Default to Anthropic if no other provider matches
    warnings.warn(f"No provider found for model '{model_name}'. Defaulting to Anthropic. "
                  f"Supported prefixes are: {list(_PROVIDERS.keys())}")
    return _PROVIDERS["claude"]


def prompt(
    prompt_text,
    model="claude-3-5-sonnet-20240620",
    system_prompt=None,
    stream=False,
    **kwargs
):
    """
    A unified function to prompt various LLMs.

    Args:
        prompt_text (str): The user's prompt.
        model (str, optional): The model to use. 
            Defaults to 'claude-3-5-sonnet-20240620'.
            Model name prefix (e.g., 'claude', 'gemini') determines the provider.
        system_prompt (str, optional): An optional system prompt.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        **kwargs: Additional provider-specific parameters.

    Returns:
        str or generator: The model's response as a string, or a generator of strings if streaming.
    """
    provider = _get_provider(model)

    # Standardize the input message format for the provider
    messages = [{"role": "user", "content": prompt_text}]

    return provider.call(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        stream=stream,
        **kwargs
    )