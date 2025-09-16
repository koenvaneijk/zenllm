import abc
from typing import Dict, Optional, List, Any

def search_pricing_data(pricing_list: List[Dict[str, Any]], model_id: str) -> Optional[Dict[str, float]]:
    """
    Finds the pricing information for a given model ID from a list of pricing data.

    Args:
        pricing_list: A list of dicts, each containing model pricing info.
        model_id: The identifier of the model.

    Returns:
        A dictionary with "input" and "output" prices per million tokens, or None if not found.
    """
    if not model_id:
        return None

    # First, try a direct match
    for model_info in pricing_list:
        if model_info.get("model_id") == model_id:
            return {
                "input": model_info["input_price_per_million_tokens"],
                "output": model_info["output_price_per_million_tokens"],
            }

    # If no direct match, try matching the last part of the ID
    simple_model_id = model_id.split('/')[-1]
    for model_info in pricing_list:
        if model_info.get("model_id") == simple_model_id:
            return {
                "input": model_info["input_price_per_million_tokens"],
                "output": model_info["output_price_per_million_tokens"],
            }

    return None


class LLMProvider(abc.ABC):
    """Abstract Base Class for all LLM providers."""

    @abc.abstractmethod
    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        """
        Makes a call to the provider's API.
        Requirements for implementers:
          - If stream=False: return a dict with keys:
              {
                "parts": [ {"type":"text","text":"..."} | {"type":"image","source":{"kind":"bytes"|"url","value":...}, "mime":"..."} ],
                "raw": <provider_raw_json>,
                "finish_reason": <str|None>,
                "usage": <dict|None>,
              }
          - If stream=True: return an iterator yielding event dicts:
              {"type":"text","text":"..."} or
              {"type":"image","bytes": b"...", "mime":"image/png"} or {"type":"image","url":"https://...","mime":"..."}
        """
        pass

    @abc.abstractmethod
    def _check_api_key(self):
        """
        Checks for the presence of the provider-specific API key.
        Must be implemented by all subclasses.
        """
        pass

    def get_model_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        """
        Finds the pricing information for a given model ID for this provider.
        Providers with pricing data should override this.

        Args:
            model_id: The identifier of the model.

        Returns:
            A dictionary with "input" and "output" prices per million tokens, or None if not found.
        """
        return None