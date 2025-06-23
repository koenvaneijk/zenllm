import abc

class LLMProvider(abc.ABC):
    """Abstract Base Class for all LLM providers."""

    @abc.abstractmethod
    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        """
        Makes a call to the provider's API.
        Must be implemented by all subclasses.
        """
        pass

    @abc.abstractmethod
    def _check_api_key(self):
        """
        Checks for the presence of the provider-specific API key.
        Must be implemented by all subclasses.
        """
        pass