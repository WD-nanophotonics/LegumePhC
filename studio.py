"""Thin launcher for the LegumePhC web Studio (localhost only).

Starts the FastAPI backend on 127.0.0.1 and opens the browser UI.
Equivalent to ``python -m legumephc.server``.
"""

from legumephc.server.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
