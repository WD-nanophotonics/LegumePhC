"""Launch the web Studio on 127.0.0.1 and open it in the default browser."""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser

import uvicorn

from .app import create_app


HOST = "127.0.0.1"


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((HOST, port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise RuntimeError("no free localhost port")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LegumePhC web Studio (localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--projects", default=None, help="Studio projects folder (default: <repo>/projects)")
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args(argv)
    port = free_port(arguments.port)
    url = f"http://{HOST}:{port}/"
    if not arguments.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"LegumePhC Studio: {url}", flush=True)
    uvicorn.run(create_app(arguments.projects), host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
