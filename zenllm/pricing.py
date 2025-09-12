from typing import Dict, Optional, List, Any
import math

# Pricing catalog
PRICING_DATA: Dict[str, List[Dict[str, Any]]] = {
  "providers": [
    {
      "provider_name": "Google Gemini",
      "models": [
        {
          "model_id": "gemini-2.5-pro",
          "input_price_per_million_tokens": 1.25,
          "output_price_per_million_tokens": 10.00
        },
        {
          "model_id": "gemini-2.5-flash",
          "input_price_per_million_tokens": 0.30,
          "output_price_per_million_tokens": 2.50
        },
        {
          "model_id": "gemini-2.5-flash-lite",
          "input_price_per_million_tokens": 0.10,
          "output_price_per_million_tokens": 0.40
        }
      ]
    },
    {
      "provider_name": "Anthropic",
      "models": [
        {
          "model_id": "claude-opus-4.1",
          "input_price_per_million_tokens": 15.00,
          "output_price_per_million_tokens": 75.00
        },
        {
          "model_id": "claude-sonnet-4",
          "input_price_per_million_tokens": 3.00,
          "output_price_per_million_tokens": 15.00
        },
        {
          "model_id": "claude-haiku-3.5",
          "input_price_per_million_tokens": 0.80,
          "output_price_per_million_tokens": 4.00
        }
      ]
    },
    {
      "provider_name": "Together.ai",
      "models": [
        {
          "model_id": "llama-3.1-405b-instruct-turbo",
          "input_price_per_million_tokens": 3.50,
          "output_price_per_million_tokens": 3.50
        },
        {
          "model_id": "deepseek-r1",
          "input_price_per_million_tokens": 3.00,
          "output_price_per_million_tokens": 7.00
        },
        {
          "model_id": "qwen3-coder-480b-a35b-instruct",
          "input_price_per_million_tokens": 2.00,
          "output_price_per_million_tokens": 2.00
        }
      ]
    },
    {
      "provider_name": "Groq",
      "models": [
        {
          "model_id": "llama-4-maverick",
          "input_price_per_million_tokens": 0.20,
          "output_price_per_million_tokens": 0.60
        },
        {
          "model_id": "moonshotai/kimi-k2-instruct-0905",
          "input_price_per_million_tokens": 1.00,
          "output_price_per_million_tokens": 3.00
        },
        {
          "model_id": "llama-3-8b-8k",
          "input_price_per_million_tokens": 0.05,
          "output_price_per_million_tokens": 0.08
        }
      ]
    },
    {
      "provider_name": "OpenAI",
      "models": [
        {
          "model_id": "gpt-5",
          "input_price_per_million_tokens": 1.25,
          "output_price_per_million_tokens": 10.00
        },
        {
          "model_id": "gpt-5-mini",
          "input_price_per_million_tokens": 0.25,
          "output_price_per_million_tokens": 2.00
        },
        {
          "model_id": "gpt-5-nano",
          "input_price_per_million_tokens": 0.05,
          "output_price_per_million_tokens": 0.40
        }
      ]
    },
    {
      "provider_name": "DeepSeek",
      "models": [
        {
          "model_id": "deepseek-chat",
          "input_price_per_million_tokens": 0.56,
          "output_price_per_million_tokens": 1.68
        },
        {
          "model_id": "deepseek-reasoner",
          "input_price_per_million_tokens": 3.00,
          "output_price_per_million_tokens": 7.00
        }
      ]
    }
  ]
}


def get_model_pricing(model_id: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Finds the pricing information for a given model ID.

    Args:
        model_id: The identifier of the model.

    Returns:
        A dictionary with "input" and "output" prices per million tokens, or None if not found.
    """
    if not model_id:
        return None

    # Some client-side model names might have prefixes (e.g., 'openai/gpt-4o' or 'together/meta-llama/Llama-3-8B-chat-hf').
    # We check for a direct match first, then try matching the part after the last '/'.
    all_models = [
        model_info
        for provider in PRICING_DATA.get("providers", [])
        for model_info in provider.get("models", [])
    ]

    # First, try a direct match
    for model_info in all_models:
        if model_info.get("model_id") == model_id:
            return {
                "input": model_info["input_price_per_million_tokens"],
                "output": model_info["output_price_per_million_tokens"],
            }

    # If no direct match, try matching the last part of the ID
    simple_model_id = model_id.split('/')[-1]
    for model_info in all_models:
        if model_info.get("model_id") == simple_model_id:
            return {
                "input": model_info["input_price_per_million_tokens"],
                "output": model_info["output_price_per_million_tokens"],
            }

    return None


def _normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """
    Normalize various provider usage dicts into:
    { 'prompt_tokens': int|None, 'completion_tokens': int|None, 'total_tokens': int|None }
    """
    if not usage:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    # OpenAI-compatible
    if any(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    # Anthropic-style
    if any(k in usage for k in ("input_tokens", "output_tokens")):
        pt = usage.get("input_tokens")
        ct = usage.get("output_tokens")
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": (pt or 0) + (ct or 0) if pt is not None and ct is not None else usage.get("total_tokens"),
        }

    # Google Gemini usageMetadata
    if any(k in usage for k in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")):
        return {
            "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        }

    # Fallback: unknown shape
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _approx_tokens_from_chars(chars: Optional[int]) -> Optional[int]:
    if chars is None:
        return None
    return int(math.ceil(chars / 4.0))


def estimate_cost(
    model: Optional[str],
    usage: Optional[Dict[str, Any]] = None,
    prompt_chars: Optional[int] = None,
    completion_chars: Optional[int] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate cost for a single request/response.
    If exact token counts are available in `usage`, uses them.
    Otherwise, falls back to approximating tokens from character counts (chars/4).
    """
    pricing = get_model_pricing(model if model else None)

    norm = _normalize_usage(usage)
    prompt_tokens: Optional[int] = norm.get("prompt_tokens")
    completion_tokens: Optional[int] = norm.get("completion_tokens")
    total_tokens: Optional[int] = norm.get("total_tokens")

    approx_used = False
    if prompt_tokens is None and prompt_chars is not None:
        prompt_tokens = _approx_tokens_from_chars(prompt_chars)
        approx_used = True
    if completion_tokens is None and completion_chars is not None:
        completion_tokens = _approx_tokens_from_chars(completion_chars)
        approx_used = True

    # Build the base structure
    result = {
        "currency": "USD",
        "model_id": model,
        "provider": provider,
        "pricing_source": "unknown" if not pricing else ("approximate" if approx_used else "known"),
        "input": {
            "tokens": prompt_tokens,
            "unit_price_per_million": pricing["input"] if pricing else None,
            "cost": None,  # filled below if possible
        },
        "output": {
            "tokens": completion_tokens,
            "unit_price_per_million": pricing["output"] if pricing else None,
            "cost": None,  # filled below if possible
        },
        "total": None,
    }

    if not pricing:
        # No pricing available; return early with tokens (if any) but without a total.
        return result

    in_price = pricing["input"]
    out_price = pricing["output"]

    in_cost = ((prompt_tokens or 0) / 1_000_000.0 * in_price) if (prompt_tokens is not None) else None
    out_cost = ((completion_tokens or 0) / 1_000_000.0 * out_price) if (completion_tokens is not None) else None

    # Set component costs
    if in_cost is not None:
        result["input"]["cost"] = round(in_cost, 6)
    if out_cost is not None:
        result["output"]["cost"] = round(out_cost, 6)

    # Compute total
    if in_cost is not None and out_cost is not None:
        result["total"] = round(in_cost + out_cost, 6)
    else:
        # If only total_tokens known and prices are equal, compute a total anyway
        if total_tokens is not None and abs(in_price - out_price) < 1e-12:
            result["total"] = round((total_tokens / 1_000_000.0) * in_price, 6)
        else:
            result["total"] = None

    return result