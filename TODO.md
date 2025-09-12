ZenLLM TODO

High-priority bugs and inconsistencies
- Fix Response.text inconsistency
  - Current class defines def text(self) but README/example use resp.text as an attribute.
  - Action: Change Response.text to a @property returning concatenated text, or add a text property delegating to the method. Update tests accordingly.
- Provider base class import path
  - Provider modules import LLMProvider from zenllm/__init__.py while it is defined in zenllm/providers/base.py.
  - Action: Update all providers to import from zenllm.providers.base to avoid circulars and to clarify ownership. Keep __init__ light.
- Ensure streaming event shape matches docs
  - README references ev.type, ev.text, ev.bytes, ev.url, ev.mime. Verify TextEvent/ImageEvent expose these attributes consistently.
  - Action: Add unit tests for ResponseStream iteration to validate event attributes and finalize() behavior.
- Timeout defaults
  - Ensure all HTTP calls use a finite timeout. RetryPolicy has a timeout field, but verify it is enforced on actual requests.
  - Action: Set explicit per-request timeouts and plumb RetryPolicy.timeout consistently across providers.

Architecture and code organization
- Reduce __init__.py surface
  - __init__ contains significant logic (providers selection, normalization, response classes). This can cause import cycles.
  - Action: Move core logic into dedicated modules (e.g., core/runtime.py, core/messages.py, core/response.py) and keep __init__ as the public API facade.
- Provider interface hardening
  - Action: Define a clear abstract base class (LLMProvider) with typed signatures, documented required behaviors (streaming semantics, usage reporting), and raise NotImplementedError for missing methods.
- Options normalization layer
  - Action: Centralize mapping from normalized options to provider-specific payloads (temperature, top_p, max_tokens, stop, tools, etc.) to avoid duplication and drift across providers.

Reliability, retries, and fallback
- Error classification audit
  - Action: Review _status_from_exception and provider-specific exception handling. Ensure 4xx vs 5xx are interpreted consistently across SDKs and raw HTTP errors.
- Retry policy with jitter
  - Action: Confirm _backoff_sleep uses decorrelated/exponential backoff with jitter. Add tests to validate backoff progression and caps.
- Mid-stream fallback lock-in
  - README says “we only lock in a provider after the first event arrives”.
  - Action: Ensure implementation adheres to allow_mid_stream_switch setting, and document exact behavior on partial stream failures (what is returned from finalize()).
- Session reuse
  - Action: Reuse HTTP sessions/clients per provider to reduce connection overhead. Consider httpx for both sync and async with proper timeouts.
- Cancellable streams
  - Action: Provide a way to cancel/close a stream early and release network resources (e.g., close generator/response). Ensure finalize() can be called after partial iteration.

Message and content handling
- Message shorthand normalization
  - Action: Add strict validation in _normalize_messages_for_chat for roles, images in system messages, and disallow malformed combinations with clear errors.
- Image handling robustness
  - Action: Centralize image source normalization, MIME detection (fallback by magic), size limits, and URL fetching (with timeouts) with clear errors and tests.
- Data URL support parity
  - Action: Ensure all OpenAI-compatible providers accept data URLs where supported; for those that don’t, convert to base64 and adapt payloads.
- System prompt routing
  - Action: Verify system_prompt is plumbed correctly per provider (OpenAI system role, Anthropic system param, Google “system_instruction”). Add tests.

Response and streaming model
- ResponseStream.finalize contract
  - Action: Define behavior when not all events were consumed, when multiple images are streamed, and when provider returns usage late. Ensure raw, usage, model, provider are set.
- Image file writing
  - Action: save_images() should infer file extension from mime when bytes; for URLs, support query-stripped extension fallback. Return list of file paths. Handle name collisions.
- Cost and usage on streaming
  - Action: Ensure usage is aggregated/available after stream finalize. If provider doesn’t report usage, estimate from observed parts as a fallback.

Pricing and cost estimation
- Pricing sources and updates
  - Action: Document and encode pricing tables with version and effective dates. Provide a mechanism to override via env or config. Add unit tests for estimate_cost.
- Token approximation
  - Action: Replace _approx_tokens_from_chars with model-specific heuristics or tiktoken/tokenizer usage when available; document approximation error.

Provider coverage and compatibility
- OpenAI-compatible routing
  - Action: Make the /chat/completions appending opt-out if base_url already includes it. Validate model-only local endpoints (Ollama/LM Studio) differences.
- Vision support detection
  - Action: Detect when a provider/model does not support vision and fail early with a clear message. Provide guidance in error text.
- Anthropic/Gemini image payloads
  - Action: Audit base64 encoding and size limits; ensure we don’t read giant files into memory without guardrails.
- X.ai, Together, DeepSeek
  - Action: Add integration tests for minimal happy-path and error-path for each. Confirm streaming deltas are parsed correctly.

API ergonomics and developer experience
- Async API
  - Action: Provide async counterparts generate_async/chat_async using httpx.AsyncClient and async generators for streaming.
- Tooling and function calling
  - Action: Plan normalized structure for tools/function calling across providers; decide on first-class support vs passthrough in options.
- Deterministic options
  - Action: Normalize temperature/top_p interplay; document defaults per provider and what happens when not supplied.

Security and privacy
- API key handling
  - Action: Ensure _check_api_key never logs key values; redact in errors. Support explicit api_key parameter overriding env safely.
- URL fetching for images
  - Action: Restrict allowed schemes (http/https), add max size, and protect against SSRF by optionally disabling remote fetches or allowing custom fetcher.
- Logging and PII
  - Action: Add structured debug logging with the ability to redact content and opt-in telemetry (disabled by default).

Documentation
- Examples accuracy
  - Action: Update examples to use resp.text as a property (post-fix), ensure code runs. Add per-provider examples, including base_url for OpenAI-compatible.
- Fallback env var docs
  - Action: Document ZENLLM_FALLBACK behavior in more detail (how options merge, how provider-level options override call-level options).
- Model mapping table
  - Action: Provide an explicit mapping of model prefixes to providers and a mechanism to override (env or config).
- Contribution guide
  - Action: Add CONTRIBUTING.md with provider-development checklist and tests requirements.

Testing and CI
- Unit tests
  - Action: Add tests for message normalization, image normalization, event streaming, finalize, cost estimation, and fallback logic.
- Integration tests
  - Action: Add optional live tests gated by env vars with recorded cassettes (pytest + vcrpy) to prevent flakiness.
- Linting and typing
  - Action: Add ruff/flake8, black, and mypy. Improve type hints (Optional, TypedDict/Protocol, Literal for providers).
- CI pipelines
  - Action: Extend GitHub Actions to run lint, type-check, unit tests, and build wheels. Cache dependencies.

Packaging and release
- pyproject.toml
  - Action: Ensure project uses pyproject.toml with pinned minimal dependencies, classifiers, long_description from README, license, and Python 3.8+ classifier.
- Versioning and changelog
  - Action: Adopt semantic versioning, add CHANGELOG.md, and automate release notes. Validate that README badges match.
- Provide py.typed
  - Action: Include a py.typed marker file and ensure type hints are part of the distribution.

Performance
- Concurrency
  - Action: Provide batch APIs or examples for concurrent calls; ensure thread-safety of global state and ResponseStream.
- Memory safety
  - Action: Avoid loading large images fully when possible; stream or chunk where provider allows.

Roadmap (nice-to-have)
- CLI
  - Action: Optional zenllm CLI for quick prompts, streaming display, and saving images.
- Caching
  - Action: Pluggable cache interface for prompts/responses with fingerprinting, disabled by default.
- Telemetry hooks
  - Action: Expose callbacks/events for request start/end, retries, cost estimation events for observability.

Housekeeping
- LICENSE and NOTICE
  - Action: Ensure MIT license file is present and included in sdist/wheel.
- Codeowners and issue templates
  - Action: Add CODEOWNERS, ISSUE_TEMPLATEs, and PR template to streamline contributions.

Quick wins (order of execution)
- Fix Response.text property and provider base import path.
- Add request timeouts and enforce RetryPolicy.timeout in all providers.
- Add basic unit tests for message normalization and ResponseStream.
- Update README examples for correctness and add a warning on vision support differences.
- Add ruff/black + mypy, configure CI to run lint and tests.