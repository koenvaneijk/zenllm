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
    print('  /model  &lt;name&gt;        Switch model (e.g., "/model gpt-4o-mini")')
    print('  /img    &lt;path(s)&gt;     Attach one or more image paths to the next user message')


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
):
    print("ZenLLM CLI — interactive chat")
    print("Type /help for commands. Press Ctrl+C or type /exit to quit.")
    print("Using model: {0}{1}".format(model, " (provider: {0})".format(provider) if provider else ""))

    messages: List[Any] = []
    if system_prompt:
        messages.append(("system", system_prompt))

    pending_images: List[str] = []
    current_model = model
    current_provider = provider
    current_system = system_prompt

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

        if user.startswith("/model "):
            new_model = user[len("/model ") :].strip()
            if new_model:
                current_model = new_model
                print("Switched model to: {0}".format(current_model))
            else:
                print("Usage: /model <model-name>")
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
    parser.add_argument("-m", "--model", default=llm.DEFAULT_MODEL, help="Model name (default from ZENLLM_DEFAULT_MODEL or {0})".format(llm.DEFAULT_MODEL))
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
    parser.add_argument("-q", "--once", default=None, help="Send a single prompt and exit (non-interactive)")

    args = parser.parse_args(argv)
    options = _build_options(args)
    stream = not args.no_stream

    # One-shot mode
    if args.once is not None:
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())