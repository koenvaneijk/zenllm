import os
import types
import unittest
from unittest.mock import patch, MagicMock

# Import the module under test
import zenllm.__init__ as z


class DummyProvider:
    """Generic dummy provider capturing calls and returning canned results."""
    def __init__(self, name_suffix=""):
        # Name affects _prov_name via class name; we dynamically build a new subclass if needed.
        self.calls = []
        self._name_suffix = name_suffix

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        self.calls.append({
            "model": model,
            "messages": messages,
            "system_prompt": system_prompt,
            "stream": stream,
            "kwargs": kwargs,
        })
        # Default behavior: return a simple text response or stream
        if stream:
            def it():
                yield {"type": "text", "text": "hello"}
            return it()
        return {
            "parts": [{"type": "text", "text": "hello"}],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "raw": {"ok": True},
        }


class FlakyProvider(DummyProvider):
    """Fails with a given exception N times, then succeeds."""
    def __init__(self, exc_sequence):
        super().__init__()
        self.exc_sequence = list(exc_sequence)  # list of Exceptions to raise sequentially, then success

    def call(self, model, messages, system_prompt=None, stream=False, **kwargs):
        self.calls.append({
            "model": model,
            "messages": messages,
            "system_prompt": system_prompt,
            "stream": stream,
            "kwargs": kwargs,
        })
        if self.exc_sequence:
            raise self.exc_sequence.pop(0)
        if stream:
            def it():
                yield {"type": "text", "text": "ok"}
            return it()
        return {
            "parts": [{"type": "text", "text": "ok"}],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "raw": {"ok": True},
        }


class ErrWithStatus(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def make_named_provider_class(name_prefix):
    """Create a new Provider class whose class name controls _prov_name."""
    return type(f"{name_prefix}Provider", (DummyProvider,), {})


class TestCoreOrchestrationAndFallback(unittest.TestCase):
    def test_generate_non_stream_propagates_and_returns_response(self):
        # Arrange a dummy provider to be returned by _get_provider
        ProvClass = make_named_provider_class("Dummy")
        dummy = ProvClass()

        def fake_get_provider(model_name, provider=None, **kwargs):
            return dummy

        with patch.object(z, "_get_provider", side_effect=fake_get_provider):
            # Act
            resp = z.generate(
                prompt="Hello world",
                model="gpt-test",
                system="sys-msg",
                stream=False,
                options={"temperature": 0.2, "foo": "bar"},
                provider="gpt",
                base_url="https://example.test",
                api_key="sk-test",
            )

        # Assert
        self.assertIsInstance(resp, z.Response)
        self.assertEqual(resp.text, "hello")
        self.assertEqual(resp.model, "gpt-test")
        self.assertEqual(resp.provider, "dummy")  # from class name DummyProvider -> "dummy"
        # Provider called exactly once
        self.assertEqual(len(dummy.calls), 1)
        call = dummy.calls[0]
        self.assertFalse(call["stream"])
        self.assertEqual(call["system_prompt"], "sys-msg")
        self.assertEqual(call["model"], "gpt-test")
        # options + base_url + api_key propagated
        self.assertIn("base_url", call["kwargs"])
        self.assertIn("api_key", call["kwargs"])
        self.assertIn("temperature", call["kwargs"])
        # message normalized
        self.assertEqual(len(call["messages"]), 1)
        msg = call["messages"][0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "Hello world")

    def test_generate_stream_returns_stream_and_finalize(self):
        ProvClass = make_named_provider_class("Dummy")
        dummy = ProvClass()

        def fake_get_provider(model_name, provider=None, **kwargs):
            return dummy

        with patch.object(z, "_get_provider", side_effect=fake_get_provider):
            resp_stream = z.generate(prompt="Hi", model="gpt-test", stream=True)

        self.assertIsInstance(resp_stream, z.ResponseStream)
        # Iterate events and finalize
        events = list(resp_stream)
        self.assertEqual(len(events), 1)
        self.assertEqual(getattr(events[0], "type", None), "text")
        self.assertEqual(getattr(events[0], "text", None), "hello")
        finalized = resp_stream.finalize()
        self.assertIsInstance(finalized, z.Response)
        self.assertEqual(finalized.text, "hello")
        # Provider was called once with stream=True
        self.assertEqual(len(dummy.calls), 1)
        self.assertTrue(dummy.calls[0]["stream"])

    def test_chat_normalizes_simple_string_and_returns_response(self):
        ProvClass = make_named_provider_class("Dummy")
        dummy = ProvClass()

        def fake_get_provider(model_name, provider=None, **kwargs):
            return dummy

        with patch.object(z, "_get_provider", side_effect=fake_get_provider):
            resp = z.chat(messages=["Hello"], model="gpt-x")

        self.assertIsInstance(resp, z.Response)
        self.assertEqual(resp.text, "hello")
        self.assertEqual(resp.model, "gpt-x")
        call = dummy.calls[0]
        self.assertEqual(call["messages"][0]["role"], "user")
        self.assertEqual(call["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(call["messages"][0]["content"][0]["text"], "Hello")

    def test_run_with_fallback_single_provider_success(self):
        # Single choice succeeds immediately
        ProvClass = make_named_provider_class("Solo")
        solo = ProvClass()

        def fake_get_provider(model_name, provider=None, **kwargs):
            return solo

        fb = z.FallbackConfig(chain=[z.ProviderChoice(provider="solo", model="m1")])

        with patch.object(z, "_get_provider", side_effect=fake_get_provider):
            resp = z._run_with_fallback(
                msgs=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                default_model="m1",
                system=None,
                stream=False,
                options=None,
                fallback=fb,
                default_provider=None,
                default_base_url=None,
                default_api_key=None,
            )

        self.assertIsInstance(resp, z.Response)
        self.assertEqual(resp.text, "hello")
        self.assertEqual(resp.finish_reason, "stop")
        self.assertEqual(resp.usage, {"prompt_tokens": 10, "completion_tokens": 3})
        # Raw fallback metadata present
        self.assertIn("fallback", resp.raw)
        self.assertEqual(resp.raw["fallback"]["selected_provider"], "solo")  # from SoloProvider -> "solo"
        self.assertEqual(resp.raw["fallback"]["selected_model"], "m1")
        self.assertEqual(resp.raw["fallback"]["attempts"], [])

    def test_retryable_failure_then_success_triggers_backoff_once(self):
        # First 429, then success
        flaky = FlakyProvider([ErrWithStatus(429)])

        def fake_get_provider(model_name, provider=None, **kwargs):
            return flaky

        fb = z.FallbackConfig(
            chain=[z.ProviderChoice(provider="flaky", model="m1")],
            retry=z.RetryPolicy(max_attempts=3, initial_backoff=0.01, max_backoff=0.02),
        )

        with patch.object(z, "_get_provider", side_effect=fake_get_provider), \
             patch.object(z.time, "sleep") as mock_sleep:
            resp = z._run_with_fallback(
                msgs=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
                default_model="m1",
                system=None,
                stream=False,
                options=None,
                fallback=fb,
                default_provider=None,
                default_base_url=None,
                default_api_key=None,
            )

        self.assertIsInstance(resp, z.Response)
        # One retry -> one sleep
        self.assertGreaterEqual(mock_sleep.call_count, 1)
        # Attempts log recorded the first failure
        attempts = resp.raw["fallback"]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0]["retryable"])

    def test_non_retryable_failure_bubbles_error(self):
        bad = FlakyProvider([ErrWithStatus(400)])  # 400 -> non-retryable

        def fake_get_provider(model_name, provider=None, **kwargs):
            return bad

        fb = z.FallbackConfig(
            chain=[z.ProviderChoice(provider="bad", model="m1")],
            retry=z.RetryPolicy(max_attempts=5, initial_backoff=0.01, max_backoff=0.02),
        )

        with patch.object(z, "_get_provider", side_effect=fake_get_provider), \
             patch.object(z.time, "sleep") as mock_sleep:
            with self.assertRaises(RuntimeError):
                z._run_with_fallback(
                    msgs=[{"role": "user", "content": [{"type": "text", "text": "x"}]}],
                    default_model="m1",
                    system=None,
                    stream=False,
                    options=None,
                    fallback=fb,
                    default_provider=None,
                    default_base_url=None,
                    default_api_key=None,
                )
        # Non-retryable -> no sleep
        mock_sleep.assert_not_called()

    def test_multi_choice_fallback_uses_next_provider(self):
        # First provider fails (non-retryable), second succeeds
        AProvider = make_named_provider_class("A")
        BProvider = make_named_provider_class("B")
        a = AProvider()
        b = BProvider()

        # Make the first provider raise during call (not during selection)
        def a_call(self, model, messages, system_prompt=None, stream=False, **kwargs):
            raise ErrWithStatus(400)
        a.call = types.MethodType(a_call, a)

        def fake_get_provider(model_name, provider=None, **kwargs):
            if provider == "A":
                return a
            return b

        fb = z.FallbackConfig(
            chain=[z.ProviderChoice(provider="A", model="ma"), z.ProviderChoice(provider="B", model="mb")],
            retry=z.RetryPolicy(max_attempts=1, initial_backoff=0.01, max_backoff=0.02),
        )

        with patch.object(z, "_get_provider", side_effect=fake_get_provider):
            resp = z._run_with_fallback(
                msgs=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                default_model="mb",
                system=None,
                stream=False,
                options=None,
                fallback=fb,
                default_provider=None,
                default_base_url=None,
                default_api_key=None,
            )

        self.assertIsInstance(resp, z.Response)
        # Selected provider should be "b"
        self.assertEqual(resp.raw["fallback"]["selected_provider"], "b")
        self.assertEqual(resp.raw["fallback"]["selected_model"], "mb")
        # Attempts log contains failure from first provider
        attempts = resp.raw["fallback"]["attempts"]
        self.assertGreaterEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["provider"], "a")  # from AProvider -> "a"


    def test_merge_options_precedence(self):
        merged = z._merge_options({"a": 1, "b": 1}, {"b": 2, "c": 3})
        self.assertEqual(merged, {"a": 1, "b": 2, "c": 3})

    def test_list_models_openai_with_base_url_and_env_key(self):
        # Fake requests.get response
        class FakeResp:
            def raise_for_status(self):
                return None
            def json(self):
                return {"data": [{"id": "m1", "created": 123, "owned_by": "org1"}]}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch.object(z.requests, "get", return_value=FakeResp()) as mock_get:
            models = z.list_models(base_url="https://mock.openai.local/v1")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "m1")
        # Ensure correct endpoint and headers used
        args, kwargs = mock_get.call_args
        self.assertTrue(args[0].endswith("/models"))
        self.assertIn("Authorization", kwargs["headers"])
        self.assertIn("Bearer", kwargs["headers"]["Authorization"])

    def test_list_models_missing_api_key_raises(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                z.list_models(base_url="https://mock.openai.local/v1")


if __name__ == "__main__":
    unittest.main()