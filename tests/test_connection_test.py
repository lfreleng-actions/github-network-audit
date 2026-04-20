# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the ``scripts/connection_test.py`` probe harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    # Import the real type for mypy; at runtime we use the dynamically
    # loaded module below so the script under test stays importable
    # from its ``scripts/`` location without a package install.
    from connection_test import Result as _Result

# The connection tester lives under ``scripts/`` rather than in the
# installed package (it is invoked directly by the testing workflow),
# so load it via ``importlib`` to make it available to pytest without
# polluting the package layout.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "connection_test.py"
_spec = importlib.util.spec_from_file_location("connection_test", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    # ``assert`` would be stripped under ``python -O``, turning this
    # into a harder-to-diagnose failure at module load time. Raise
    # ImportError so the failure mode is explicit and runs under any
    # optimisation level.
    raise ImportError(f"Unable to load module spec for {_SCRIPT_PATH}")
connection_test = importlib.util.module_from_spec(_spec)
sys.modules["connection_test"] = connection_test
_spec.loader.exec_module(connection_test)

Result = connection_test.Result
Target = connection_test.Target
_positive_float = connection_test._positive_float
_positive_int = connection_test._positive_int
_probe_once = connection_test._probe_once
parse_targets = connection_test.parse_targets
render_summary = connection_test.render_summary


class TestParseTargets:
    """Unit tests for ``parse_targets``."""

    def test_simple_newline_separated(self) -> None:
        """Parses one entry per line."""
        targets = parse_targets("api.github.com:443\npypi.org:443\n")
        assert targets == [
            Target(host="api.github.com", port=443),
            Target(host="pypi.org", port=443),
        ]

    def test_tolerates_blank_lines_and_crlf(self) -> None:
        """Blank lines and CRLF endings are ignored."""
        raw = "api.github.com:443\r\n\r\npypi.org:443\r\n"
        assert parse_targets(raw) == [
            Target(host="api.github.com", port=443),
            Target(host="pypi.org", port=443),
        ]

    def test_ignores_full_line_comments(self) -> None:
        """A leading ``#`` marks the whole line as a comment."""
        raw = "# header comment\napi.github.com:443\n"
        assert parse_targets(raw) == [Target(host="api.github.com", port=443)]

    def test_ignores_inline_comments(self) -> None:
        """Trailing ``# comment`` on an entry line is stripped."""
        raw = "api.github.com:443  # primary endpoint\npypi.org:443#pkg\n"
        assert parse_targets(raw) == [
            Target(host="api.github.com", port=443),
            Target(host="pypi.org", port=443),
        ]

    def test_space_separated_tokens(self) -> None:
        """Space-separated entries on a single line are supported."""
        targets = parse_targets("a.example:443 b.example:80")
        assert targets == [
            Target(host="a.example", port=443),
            Target(host="b.example", port=80),
        ]

    def test_preserves_duplicates(self) -> None:
        """Duplicates are not collapsed; the caller decides policy."""
        targets = parse_targets("example.com:443\nexample.com:443\n")
        assert len(targets) == 2

    def test_rejects_missing_port(self) -> None:
        """A token without ``:port`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="missing ':port'"):
            parse_targets("example.com\n")

    def test_rejects_non_numeric_port(self) -> None:
        """A non-integer port raises ``ValueError``."""
        with pytest.raises(ValueError, match="invalid port"):
            parse_targets("example.com:abc\n")

    def test_rejects_out_of_range_port(self) -> None:
        """Ports outside 1..65535 are rejected."""
        with pytest.raises(ValueError, match="port out of range"):
            parse_targets("example.com:70000\n")

    def test_rejects_port_zero(self) -> None:
        """Port 0 is rejected."""
        with pytest.raises(ValueError, match="port out of range"):
            parse_targets("example.com:0\n")


class TestRenderSummary:
    """Unit tests for ``render_summary``."""

    @staticmethod
    def _make_results(
        *specs: tuple[str, int, bool, int, str | None],
    ) -> list[_Result]:
        """Build ``Result`` objects from ``(host, port, connected, attempts, error)``."""
        return [
            Result(
                target=Target(host=h, port=p),
                connected=c,
                attempts=a,
                error=e,
            )
            for h, p, c, a, e in specs
        ]

    def test_permitted_all_passing(self) -> None:
        """A fully-passing permitted run reports 0 failures."""
        results = self._make_results(
            ("api.github.com", 443, True, 1, None),
            ("pypi.org", 443, True, 1, None),
        )
        text, failures = render_summary(results, mode="permitted", connect_timeout=5.0, max_attempts=3)
        assert failures == 0
        assert "**Failures:** 0 / 2" in text
        assert "connected" in text
        assert "## Test PERMITTED Connections" in text

    def test_permitted_with_failure(self) -> None:
        """A failing permitted probe increments the failure count."""
        results = self._make_results(
            ("api.github.com", 443, True, 1, None),
            ("unreachable.example", 443, False, 3, "timeout"),
        )
        text, failures = render_summary(results, mode="permitted", connect_timeout=5.0, max_attempts=3)
        assert failures == 1
        assert "**Failures:** 1 / 2" in text
        assert "FAILED (timeout)" in text

    def test_denied_all_blocked(self) -> None:
        """A fully-blocked denied run reports 0 leaks."""
        results = self._make_results(
            ("www.example.org", 443, False, 1, "refused"),
            ("www.wikipedia.org", 443, False, 1, "refused"),
        )
        text, failures = render_summary(results, mode="denied", connect_timeout=5.0, max_attempts=1)
        assert failures == 0
        assert "**Leaks:** 0 / 2" in text
        assert "blocked (refused)" in text
        assert "## Test DENIED Connections" in text

    def test_denied_with_leak(self) -> None:
        """A connected denied probe counts as a leak."""
        results = self._make_results(
            ("leaky.example", 443, True, 1, None),
            ("www.example.org", 443, False, 1, "refused"),
        )
        text, failures = render_summary(results, mode="denied", connect_timeout=5.0, max_attempts=1)
        assert failures == 1
        assert "**Leaks:** 1 / 2" in text
        assert "LEAKED (connected)" in text

    def test_duplicates_are_preserved_in_totals(self) -> None:
        """Duplicate host:port entries appear twice in the totals."""
        results = self._make_results(
            ("example.com", 443, True, 1, None),
            ("example.com", 443, True, 1, None),
        )
        text, failures = render_summary(results, mode="permitted", connect_timeout=5.0, max_attempts=3)
        assert failures == 0
        # Count of 2 proves duplicates are not deduplicated.
        assert "**Failures:** 0 / 2" in text
        assert text.count("`example.com:443`") == 2

    def test_denied_incomplete_probe_counts_as_leak(self) -> None:
        """An overall-timeout in denied mode is a failure, not a pass.

        A denied probe that never ran (``attempts == 0``, typically
        because the overall wall-clock cap expired before the task
        finished) leaves the endpoint untested; treating it as
        "blocked" would mask silent misconfigurations and let a run
        with an aggressive ``--overall-timeout`` exit 0 without
        actually testing anything.
        """
        results = self._make_results(
            ("www.example.org", 443, False, 0, "overall-timeout"),
            ("www.wikipedia.org", 443, False, 1, "refused"),
        )
        text, failures = render_summary(results, mode="denied", connect_timeout=5.0, max_attempts=1)
        assert failures == 1
        assert "**Leaks:** 1 / 2" in text
        assert "INCOMPLETE (overall-timeout)" in text
        assert "blocked (refused)" in text


class TestProbeError:
    """Unit tests for ``_probe_once``'s error-formatting logic."""

    def test_refused_error_message_has_no_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused connect reports a canonical errno message.

        asyncio's own ``str(exc)`` formatting embeds the resolved IP
        address for connect failures. Under harden-runner every denied
        target resolves to the same sinkhole address, which made every
        row of the denied-mode summary look identical and suggested a
        bug. Simulate a refused connect by patching
        ``asyncio.open_connection`` (so the test is deterministic on
        any platform and does not depend on whether ``127.0.0.1:1`` is
        actually open) and assert that the error string contains
        neither the address nor the port.
        """
        import asyncio
        import errno

        async def _fake_refused(*_a: object, **_k: object) -> None:
            # Emulate a real ConnectionRefusedError whose ``str(exc)``
            # embeds the peer address, just like asyncio's wrapper.
            raise ConnectionRefusedError(
                errno.ECONNREFUSED,
                "Connect call failed ('1.2.3.4', 443)",
            )

        monkeypatch.setattr(connection_test.asyncio, "open_connection", _fake_refused)

        connected, error = asyncio.run(_probe_once(Target("peer.example", 443), 2.0))
        assert connected is False
        assert error is not None
        assert "1.2.3.4" not in error
        assert ":443" not in error
        # Compare against os.strerror directly so the test is not
        # sensitive to the host's locale (which could translate the
        # English text) nor to minor wording differences across
        # platforms.
        import os

        assert error == os.strerror(errno.ECONNREFUSED)

    def test_resolver_failure_keeps_synthesised_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative EAI_* errno falls back to the synthesised message.

        ``os.strerror()`` returns "Unknown error -N" for the negative
        ``EAI_*`` codes used by ``socket.gaierror``, so those paths
        must preserve the exception's strerror / str(exc) text, which
        for resolver failures does not embed an IP because DNS
        resolution failed before a connect was attempted.
        """
        import asyncio
        import socket

        async def _fake_gaierror(*_a: object, **_k: object) -> None:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(connection_test.asyncio, "open_connection", _fake_gaierror)

        connected, error = asyncio.run(_probe_once(Target("no-such-host.example", 443), 2.0))
        assert connected is False
        assert error == "Name or service not known"


class TestArgValidators:
    """Unit tests for the ``argparse`` type validators."""

    def test_positive_float_accepts_positive(self) -> None:
        """A positive numeric string round-trips to float."""
        assert _positive_float("2.5") == 2.5

    def test_positive_float_rejects_zero(self) -> None:
        """Zero is out of range for a strictly positive float."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
            _positive_float("0")

    def test_positive_float_rejects_negative(self) -> None:
        """Negative values are out of range."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
            _positive_float("-1")

    def test_positive_float_rejects_non_numeric(self) -> None:
        """A non-numeric value produces an argparse error."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="expected a number"):
            _positive_float("abc")

    def test_positive_int_accepts_one(self) -> None:
        """One is the minimum valid attempt count."""
        assert _positive_int("1") == 1

    def test_positive_int_rejects_zero(self) -> None:
        """Zero would produce a zero-iteration probe loop."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
            _positive_int("0")

    def test_positive_int_rejects_non_integer(self) -> None:
        """Non-integer values are rejected before validation."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
            _positive_int("1.5")
