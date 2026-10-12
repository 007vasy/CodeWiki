"""Tests for the unswallowable per-file parse timeout and asset ignoring."""

import time
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

from codewiki.src.be.dependency_analyzer.analysis.call_graph_analyzer import (
    FileParseTimeout,
    timeout,
)
from codewiki.src.be.dependency_analyzer.analysis.repo_analyzer import (
    RepoAnalyzer,
)
from codewiki.src.be.dependency_analyzer.utils.patterns import DEFAULT_IGNORE_PATTERNS


def test_timeout_raises_through_broad_except_exception():
    """The old TimeoutError (an Exception) was swallowed by analyzers' broad
    ``except Exception`` guards and a pathological file ground for hours
    inside the per-file cap. FileParseTimeout must sail through them."""

    def pathological():
        # Mimics a language analyzer's inner loop with a broad guard.
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                for _ in range(10_000_000):
                    pass
            except Exception:  # noqa: BLE001 — the bug being pinned down
                continue  # previously swallowed the alarm here

    with pytest.raises(FileParseTimeout):
        with timeout(1):
            pathological()


def test_timeout_restores_alarm_and_handler():
    assert "SIGALRM" in dir(__import__("signal"))  # sanity: unix handler path

    def slow():
        time.sleep(3)

    with pytest.raises(FileParseTimeout):
        with timeout(1):
            slow()
    # The alarm is cancelled in the finally: a fresh slow body must run to
    # completion (not be killed by a stale alarm).
    with timeout(10):
        slow()


def test_committed_web_assets_are_ignored():
    assert "assets" in DEFAULT_IGNORE_PATTERNS
    analyzer = RepoAnalyzer()
    # The walk prunes the assets directory itself, like node_modules.
    assert analyzer._should_exclude_path("web/assets", "assets", True)
    assert analyzer._should_exclude_path("assets", "assets", True)
