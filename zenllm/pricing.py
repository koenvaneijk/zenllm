from typing import Dict, Optional, Any
import math


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
    provider_name: Optional[str] = None,
    provider_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate cost for a single request/response.
    If exact token counts are available in `usage`, uses them.
    Otherwise, falls back to approximating tokens from character counts (chars/4).
    """
    pricing = None
    if provider_name and model and provider_map:
        provider_instance = provider_map.get(provider_name.lower())
        if provider_instance:
            pricing = provider_instance.get_model_pricing(model)

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
        "provider": provider_name,
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