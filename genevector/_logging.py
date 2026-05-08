"""Central logging configuration for GeneVector.

Exposes ``get_logger(name)`` for module-level loggers. Configures the
``genevector`` parent logger lazily on first use with either RichHandler
(TTY) or plain StreamHandler (non-TTY). Level controlled via the
``GENEVECTOR_LOG_LEVEL`` env var (default ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys

_configured = False
_console = None


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _configure_root() -> None:
    global _configured, _console
    if _configured:
        return

    level_name = os.environ.get("GENEVECTOR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("genevector")
    root.setLevel(level)
    root.propagate = False

    if root.handlers:
        _configured = True
        return

    handler: logging.Handler
    if _is_tty():
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            _console = Console()
            handler = RichHandler(
                console=_console,
                show_time=False,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured under the ``genevector`` parent."""
    _configure_root()
    return logging.getLogger(name)


def get_console():
    """Return the shared rich Console, or None if running in plain mode.

    For internal use by future tasks (panels, progress bars). Most callers
    should use ``get_logger()`` instead.
    """
    _configure_root()
    return _console
