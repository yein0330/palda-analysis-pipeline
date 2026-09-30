#!/usr/bin/env python3
"""Serve one review directory over temporary HTTP Basic authentication."""

from __future__ import annotations

import argparse
import base64
import secrets
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--username", default="reviewer")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    directory = args.directory.resolve(strict=True)
    password = secrets.token_urlsafe(15)
    expected = "Basic " + base64.b64encode(
        f"{args.username}:{password}".encode("utf-8")
    ).decode("ascii")

    class AuthenticatedHandler(SimpleHTTPRequestHandler):
        def do_HEAD(self) -> None:
            if self._authenticated():
                super().do_HEAD()

        def do_GET(self) -> None:
            if self._authenticated():
                super().do_GET()

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

        def _authenticated(self) -> bool:
            if secrets.compare_digest(self.headers.get("Authorization", ""), expected):
                return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="PALDA Review"')
            self.end_headers()
            return False

        def log_message(self, format: str, *values: object) -> None:
            # Avoid logging paths that can contain review identifiers.
            print(f"request from {self.client_address[0]}: {values[1]}", flush=True)

    handler = partial(AuthenticatedHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"PALDA_REVIEW_URL=http://{args.bind}:{args.port}/review.html", flush=True)
    print(f"PALDA_REVIEW_USERNAME={args.username}", flush=True)
    print(f"PALDA_REVIEW_PASSWORD={password}", flush=True)
    print("Press Ctrl+C to stop the temporary review server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
