"""Tests for genevector._logging."""

from __future__ import annotations

import logging
import os

import pytest

import genevector._logging as _gv_logging
from genevector._logging import get_logger


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset module-level configured flag and parent logger handlers between tests."""
    _gv_logging._configured = False
    _gv_logging._console = None
    parent = logging.getLogger("genevector")
    parent.handlers.clear()
    parent.setLevel(logging.NOTSET)
    yield
    _gv_logging._configured = False
    _gv_logging._console = None
    parent.handlers.clear()
    parent.setLevel(logging.NOTSET)


def test_get_logger_returns_logger():
    log = get_logger("genevector.data")
    assert isinstance(log, logging.Logger)
    assert log.name == "genevector.data"


def test_logger_level_from_env(monkeypatch):
    monkeypatch.setenv("GENEVECTOR_LOG_LEVEL", "WARNING")
    get_logger("genevector.data")
    assert logging.getLogger("genevector").level == logging.WARNING


def test_logger_level_default_info(monkeypatch):
    monkeypatch.delenv("GENEVECTOR_LOG_LEVEL", raising=False)
    get_logger("genevector.data")
    assert logging.getLogger("genevector").level == logging.INFO


def test_handler_idempotent():
    for _ in range(5):
        get_logger("genevector.data")
    assert len(logging.getLogger("genevector").handlers) == 1


def test_no_propagation_to_root():
    get_logger("genevector.data")
    assert logging.getLogger("genevector").propagate is False
