import argparse
import sys
from typing import Any, Dict, List, Optional

import zenllm as llm


def _build_options(args) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    if args.temperature is not None:
        opts["temperature"] = args.temperature
    if args.top_p is not None:
        opts["top_p"] = args.top_p
    if args.max_tokens is not None:
        opts["max_tokens"] = args.max_tokens
    return opts


def _print_help_commands():
    print("Commands:")
    print("  /help                 Show this help")
    print("  /exit | /quit | :q    Exit the chat")
    print("  /reset                Reset conversation history")
    print('  /system &lt;text&gt;       Set/replace the system prompt for the session')
    print('  /model  [name]        Switch model (omit name to select interactively)')
    print('  /img    &lt;path(s)&gt;     Attach one or more image paths to the next user message')


def _select_model_interactive(
    provider: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    limit: int = 200,
) -> Optional[str]:
    """
    Interactively select a model.
    - Tries to list models for OpenAI-compatible providers (or base_url).
    - Falls back to manual input if listing is not supported.
    Returns selected model id, a manually entered name, or None on cancel.
    For OpenAI ("openai" or "gpt" provider), pressing Enter selects "gpt-5".
    """
    is_openai = bool(provider and provider.lower() in ("openai", "gpt"))

    print("Model selection:")
    models: List[str] = []
    try:
        listed = llm.list_models(provider=provider, base_url=base_url, api_key=api_key)
        models = sorted([m.id for m in listed])
    except NotImplementedError:
        print("Listing models is not supported for this provider. Enter a model name manually.")
    except Exception as e:
        print(f"Could not fetch model list: {e}")
        print("Enter a model name manually or press Enter to {('select gpt-5' if is_openai else 'cancel')}.")

    if models:
        if limit and len(models) > limit:
            print(f"Fetched {len(models)} models; showing first {limit}. Use a name to select hidden ones.")
            display = models[:limit]
        else:
            display = models
        for i, mid in enumerate(display, 1):
            print(f"  {i:3d}. {mid}")

        while True:
            prompt_str = "Select model (# or name, empty for gpt-5): " if is_openai else "Select model (# or name, empty to cancel): "
            sel = input(prompt_str).strip()
            if not sel:
                return "gpt-5" if is_openai else None
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(display):
                    return display[idx - 1]
                print(f"Enter a number between 1 and {len(display)}, a model name, or press Enter to {'select gpt-5' if is_openai else 'cancel'}.")
                continue
            # Accept direct model name
            return sel

    # Fallback: manual entry
    manual_prompt = "Model name (empty for gpt-5): " if is_openai else "Model name (empty to cancel): "
    manual = input(manual_prompt).strip()
    return manual or ("gpt-5" if is_openai else None)


def _interactive_chat(
    model: str,
    provider: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    system_prompt: Optional[str],
    stream: bool,
    show_usage: bool,
    show_cost: bool,
    options: Dict[str, Any],
    select_model: bool = False,
):
    print("ZenLLM CLI — interactive chat")
    print("Type /help for commands. Press Ctrl+C or type /exit to quit.")

    messages: List[Any] = []
    if system_prompt:
        messages.append(("system", system_prompt))

    pending_images: List[str] = []
    current_model = model
    current_provider = provider
    current_system = system_prompt

    # Optional pre-session model selection
    if select_model:
        chosen = _select_model_interactive(current_provider, base_url, api_key)
        if chosen:
            current_model = chosen
            print("Selected model: {0}".format(current_model))

    print("Using model: {0}{1}".format(current_model, " (provider: {0})".format(current_provider) if current_provider else ""))

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue

        if user in ("/exit", "/quit", ":q", "q"):
            break

        if user.startswith("/help"):
            _print_help_commands()
            continue

        if user.startswith("/reset"):
            messages = []
            if current_system:
                messages.append(("system", current_system))
            pending_images = []
            print("Conversation reset.")
            continue

        if user.startswith("/system "):
            current_system = user[len("/system ") :].strip() or None
            # Replace or insert system message at the beginning
            if messages and isinstance(messages[0], tuple) and messages[0][0] == "system":
                messages[0] = ("system", current_system or "")
            else:
                if current_system:
                    messages.insert(0, ("system", current_system))
            print("System prompt set.")
            continue

        if user.startswith("/model"):
            arg = user[len("/model") :].strip()
            if not arg:
                chosen = _select_model_interactive(current_provider, base_url, api_key)
                if chosen:
                    current_model = chosen
                    print("Switched model to: {0}".format(current_model))
                else:
                    print("Model unchanged.")
            else:
                current_model = arg
                print("Switched model to: {0}".format(current_model))
            continue

        if user.startswith("/img "):
            paths = user[len("/img ") :].strip().split()
            if not paths:
                print("Usage: /img <path1> [path2 ...]")
                continue
            pending_images.extend(paths)
            print("Attached {0} image(s) to the next message.".format(len(pending_images)))
            continue

        # Regular user message
        msg_images = pending_images if pending_images else None
        messages.append(("user", user, msg_images))
        pending_images = []

        try:
            if stream:
                rs = llm.chat(
                    messages,
                    model=current_model,
                    system=current_system,
                    stream=True,
                    options=options,
                    provider=current_provider,
                    base_url=base_url,
                    api_key=api_key,
                )
                # Stream tokens
                for ev in rs:
                    if ev.type == "text":
                        print(ev.text, end="", flush=True)
                resp = rs.finalize()
                print()
            else:
                resp = llm.chat(
                    messages,
                    model=current_model,
                    system=current_system,
                    stream=False,
                    options=options,
                    provider=current_provider,
                    base_url=base_url,
                    api_key=api_key,
                )
                print(resp.text)

            # Append assistant turn for context
            if resp.text:
                messages.append(("assistant", resp.text))

            if show_usage and resp.usage:
                print("usage:", resp.usage)
            if show_cost:
                cost = resp.cost()
                if cost is not None:
                    print("cost: ${0:.6f}".format(cost))
        except KeyboardInterrupt:
            print("\n(Interrupted)")
            continue
        except Exception as e:
            print("Error: {0}".format(e), file=sys.stderr)

    print("Goodbye.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zenllm",
        description="Chat with LLMs in your terminal using ZenLLM.",
    )
    parser.add_argument("-m", "--model", default=None, help="Model name (default: gpt-5 for provider openai/gpt; otherwise ZENLLM_DEFAULT_MODEL or {0})".format(llm.DEFAULT_MODEL))
    parser.add_argument("--provider", default=None, help="Force provider (e.g., openai, gemini, claude, deepseek, together, xai, groq)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (e.g., http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=None, help="Override API key (otherwise use provider-specific env var)")
    parser.add_argument("-s", "--system", default=None, help="System prompt for the session")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--show-usage", action="store_true", help="Print usage dict after each response (if available)")
    parser.add_argument("--show-cost", action="store_true", help="Print cost estimate after each response (if pricing available)")
    parser.add_argument("--select-model", action="store_true", help="Interactively select a model from the provider (OpenAI-compatible endpoints)")
    parser.add_argument("-q", "--once", default=None, help="Send a single prompt and exit (non-interactive)")

    arg_list = list(argv) if argv is not None else sys.argv[1:]
    # Detect whether the user explicitly provided --model/-m
    user_provided_model = False
    for tok in arg_list:
        if tok in ("-m", "--model") or tok.startswith("--model="):
            user_provided_model = True
            break

    args = parser.parse_args(arg_list)
    options = _build_options(args)
    stream = not args.no_stream

    # If no model provided, choose defaults:
    # - For OpenAI ("openai" or "gpt" provider): use "gpt-5"
    # - Otherwise: use library default (env ZENLLM_DEFAULT_MODEL or {0})
    if args.model is None:
        if args.provider and args.provider.lower() in ("openai", "gpt"):
            args.model = "gpt-5"
        else:
            args.model = llm.DEFAULT_MODEL

    # Determine if we should prompt for model selection (default when --model was not provided)
    auto_select_model = args.select_model or (not user_provided_model)

    # One-shot mode
    if args.once is not None:
        # Optional pre-selection (default when --model was not provided)
        if auto_select_model:
            chosen = _select_model_interactive(args.provider, args.base_url, args.api_key)
            if chosen:
                args.model = chosen

        msgs: List[Any] = []
        if args.system:
            msgs.append(("system", args.system))
        msgs.append(("user", args.once))
        try:
            if stream:
                rs = llm.chat(
                    msgs,
                    model=args.model,
                    system=args.system,
                    stream=True,
                    options=options,
                    provider=args.provider,
                    base_url=args.base_url,
                    api_key=args.api_key,
                )
                for ev in rs:
                    if ev.type == "text":
                        print(ev.text, end="", flush=True)
                resp = rs.finalize()
                print()
            else:
                resp = llm.chat(
                    msgs,
                    model=args.model,
                    system=args.system,
                    stream=False,
                    options=options,
                    provider=args.provider,
                    base_url=args.base_url,
                    api_key=args.api_key,
                )
                print(resp.text)

            if args.show_usage and resp.usage:
                print("usage:", resp.usage)
            if args.show_cost:
                cost = resp.cost()
                if cost is not None:
                    print("cost: ${0:.6f}".format(cost))
            return 0
        except Exception as e:
            print("Error: {0}".format(e), file=sys.stderr)
            return 1

    # Interactive chat mode
    _interactive_chat(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system,
        stream=stream,
        show_usage=args.show_usage,
        show_cost=args.show_cost,
        options=options,
        select_model=auto_select_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
