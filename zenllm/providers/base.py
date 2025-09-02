import abc

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