from __future__ import annotations

import logging

import pytest

from shamela_rag.logging_config import configure_logging, get_logger


def test_configure_sets_level() -> None:
    configure_logging(level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_get_logger_emits(caplog: pytest.LogCaptureFixture) -> None:
    log = get_logger("shamela_rag.test")
    with caplog.at_level(logging.INFO):
        log.info("hello world")
    assert "hello world" in caplog.text
