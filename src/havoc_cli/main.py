from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI client for HAVOC-7B inference server")
    parser.add_argument("prompt", type=str, help="Prompt to send to the model")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Inference server base URL")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming; wait for full response")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    args = parser.parse_args()

    body = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": True,
        "stream": not args.no_stream,
    }

    endpoint = urllib.parse.urljoin(args.host.rstrip("/") + "/", "completion")
    try:
        if body["stream"]:
            for token in stream_request(endpoint, body):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()
        else:
            text = non_stream_request(endpoint, body)
            print(text)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        detail = payload.decode("utf-8", errors="ignore")
        parser.error(f"Server error ({exc.code}): {detail}")
    except Exception as exc:
        parser.error(f"Request failed: {exc}")


def stream_request(url: str, body: dict) -> Iterator[str]:
    """Stream tokens from the inference server using SSE."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            yield data


def non_stream_request(url: str, body: dict) -> str:
    """Send a non-streaming completion request."""
    body = {**body, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    return parsed.get("text", "")


if __name__ == "__main__":
    main()
