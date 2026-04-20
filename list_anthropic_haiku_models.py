#!/usr/bin/env python3
"""List available Anthropic model IDs filtered by a pattern (default: haiku).

Uses the Anthropic Models list endpoint:
  GET https://api.anthropic.com/v1/models
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List available Anthropic model IDs (default filter: haiku)."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.anthropic.com",
        help="Anthropic API base URL (default: https://api.anthropic.com)",
    )
    parser.add_argument(
        "--contains",
        default="haiku",
        help="Case-insensitive substring filter applied to model id/display name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print matching model objects as JSON instead of only IDs.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (not recommended).",
    )
    return parser.parse_args()


def load_api_key(explicit_key: str | None) -> str:
    if explicit_key:
        return explicit_key.strip()

    # Optional .env loading for local workflows.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing API key. Set ANTHROPIC_API_KEY or pass --api-key."
        )
    return key


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_models(base_url: str, api_key: str, insecure: bool) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/models"
    req = Request(
        url,
        method="GET",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=30, context=_ssl_context(insecure)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc

    models = payload.get("data", [])
    if not isinstance(models, list):
        raise RuntimeError("Unexpected response format from /v1/models")
    return [m for m in models if isinstance(m, dict)]


def is_match(model_obj: dict[str, Any], pattern: str) -> bool:
    p = pattern.lower()
    model_id = str(model_obj.get("id", "")).lower()
    display_name = str(model_obj.get("display_name", "")).lower()
    return p in model_id or p in display_name


def main() -> int:
    args = parse_args()
    try:
        api_key = load_api_key(args.api_key)
        models = fetch_models(args.base_url, api_key, args.insecure)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    matches = [m for m in models if is_match(m, args.contains)]

    if args.json:
        print(json.dumps(matches, indent=2, ensure_ascii=False))
    else:
        if not matches:
            print(f"No models matched filter: {args.contains}")
            return 0
        print(f"Matched models ({len(matches)}):")
        for m in matches:
            model_id = str(m.get("id", "")).strip()
            display_name = str(m.get("display_name", "")).strip()
            if display_name:
                print(f"- {model_id}\t({display_name})")
            else:
                print(f"- {model_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
