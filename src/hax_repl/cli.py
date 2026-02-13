from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hax_repl.repl import run_repl
from hax_repl.runtime import AppRuntime


def configure_logging() -> Path:
    logs_dir = Path.home() / ".local" / "share" / "haxllm" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "hax-repl.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    return log_file


def main() -> None:
    parser = argparse.ArgumentParser(description="HAX LLM REPL")
    parser.add_argument("--session", type=str, default=None, help="Session name")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="OpenRouter model name",
    )
    args = parser.parse_args()

    log_file = configure_logging()

    runtime = AppRuntime(session_name=args.session, model_name=args.model)
    logging.getLogger(__name__).info("Starting hax-repl session=%s log_file=%s", args.session, log_file)
    run_repl(runtime)


if __name__ == "__main__":
    main()
