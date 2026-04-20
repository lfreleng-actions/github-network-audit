#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
"""Connection allowlist validator.

Parallel TCP connect tests for GitHub Actions hardening workflows.

The script reads a whitespace-separated list of ``host:port`` targets
(one per line is the conventional form, but spaces or tabs also work)
from an environment variable and probes each one concurrently using
``asyncio``. Two modes are supported:

* ``permitted`` — targets are expected to connect. Each target is
  retried up to ``--max-attempts`` times with ``--connect-timeout``
  seconds per attempt. A failure to connect is a test failure.
* ``denied`` — targets are expected to be blocked. A single attempt is
  made per target (retries would only slow down the wall-clock). A
  successful connection is a test failure (the allowlist leaked).

Why a Python script rather than inline bash?

* Cancellation is instantaneous: ``asyncio`` wait-groups plus SIGTERM
  / SIGINT handlers tear down every in-flight probe when the runner
  cancels the step. Background bash probes under ``timeout`` routinely
  outlive their parent shell, which is why the previous workflow
  iteration kept hanging on cancel.
* Wall-clock is bounded deterministically. ``--overall-timeout`` caps
  the entire probe phase; results collected so far are still rendered
  to ``$GITHUB_STEP_SUMMARY`` and the exit status reflects the verdict.
* Expected pass/fail counts derive from the number of parsed target
  tokens (typically one per input line), so the job scales as you add
  or remove endpoints from the variable without touching the workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["permitted", "denied"]


@dataclass(frozen=True)
class Target:
    """A parsed ``host:port`` target."""

    host: str
    port: int

    @property
    def label(self) -> str:
        """Return the canonical ``host:port`` label."""
        return f"{self.host}:{self.port}"


@dataclass
class Result:
    """Outcome of probing a single target."""

    target: Target
    connected: bool
    attempts: int
    error: str | None


def parse_targets(raw: str) -> list[Target]:
    """Parse a whitespace-separated list of ``host:port`` entries.

    Blank lines, full-line comments (``#``) and inline trailing
    comments (``host:port  # comment``) are tolerated so the variable
    can be edited comfortably in the GitHub Actions UI. Surrounding
    whitespace is ignored.
    """

    targets: list[Target] = []
    for raw_line in raw.replace("\r", "\n").split("\n"):
        # Strip inline comments before tokenising so a trailing
        # ``# note`` on the same line does not confuse the parser.
        line = raw_line.split("#", 1)[0]
        for token in line.split():
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"missing ':port' in target: {token!r}")
            host, _, port_str = token.rpartition(":")
            if not host or not port_str:
                raise ValueError(f"invalid target: {token!r}")
            try:
                port = int(port_str)
            except ValueError as exc:
                raise ValueError(f"invalid port in target {token!r}: {port_str!r}") from exc
            if not 1 <= port <= 65535:
                raise ValueError(f"port out of range in target {token!r}: {port}")
            targets.append(Target(host=host, port=port))
    return targets


async def _probe_once(target: Target, timeout: float) -> tuple[bool, str | None]:
    """Perform a single TCP connect. Returns (connected, error)."""

    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target.host, target.port),
            timeout=timeout,
        )
    except TimeoutError:
        return False, "timeout"
    except OSError as exc:
        # asyncio synthesises ``strerror`` / ``str(exc)`` messages that
        # embed the resolved IP address (e.g. "Connect call failed
        # ('54.185.253.63', 443)"). For denied endpoints that IP is just
        # the harden-runner sinkhole, which makes every row of the
        # summary look identical and suggests a bug. Prefer the canonical
        # ``os.strerror(errno)`` (e.g. "Connection refused") for normal
        # positive socket errno values. Resolver failures (typically
        # ``socket.gaierror``) use negative ``EAI_*`` codes for which
        # ``os.strerror()`` returns unhelpful text like
        # "Unknown error -2"; fall back to the synthesised message for
        # those, which does not embed an IP because resolution failed
        # before a connect was attempted.
        if exc.errno is not None:
            if exc.errno > 0:
                return False, os.strerror(exc.errno)
            if exc.errno < 0:
                return False, exc.strerror or str(exc) or type(exc).__name__
        return False, type(exc).__name__
    else:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            # A half-open transport failing to close cleanly does not
            # invalidate the fact that the connect succeeded; log
            # nothing and return success.
            pass
        # Intentionally do not catch ``asyncio.CancelledError`` here:
        # suppressing it would clear the task's cancelled state and
        # let a probe report "connected" even though the runner is
        # tearing the step down. Let it propagate so the event loop
        # honours the cancellation quickly.
        return True, None


async def probe_target(
    target: Target,
    *,
    mode: Mode,
    connect_timeout: float,
    max_attempts: int,
) -> Result:
    """Probe ``target`` according to ``mode`` and return the outcome."""

    # Denied targets should never connect; a single attempt is enough
    # and avoids slowing the job down waiting for multiple timeouts on
    # a correctly-blocked endpoint.
    attempts_allowed = 1 if mode == "denied" else max_attempts
    last_error: str | None = None
    for attempt in range(1, attempts_allowed + 1):
        connected, error = await _probe_once(target, connect_timeout)
        if connected:
            return Result(target=target, connected=True, attempts=attempt, error=None)
        last_error = error
    return Result(
        target=target,
        connected=False,
        attempts=attempts_allowed,
        error=last_error,
    )


async def run_probes(
    targets: list[Target],
    *,
    mode: Mode,
    connect_timeout: float,
    max_attempts: int,
    overall_timeout: float,
) -> list[Result]:
    """Probe all targets concurrently under a global wall-clock cap."""

    tasks = [
        asyncio.create_task(
            probe_target(
                target,
                mode=mode,
                connect_timeout=connect_timeout,
                max_attempts=max_attempts,
            ),
            name=f"probe:{target.label}",
        )
        for target in targets
    ]
    try:
        done = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=False),
            timeout=overall_timeout,
        )
    except TimeoutError:
        # The overall wall-clock cap expired. Collect whatever has
        # completed; distinguish cleanly-finished tasks, tasks that
        # raised, and tasks still running so a real bug in
        # ``probe_target()`` is surfaced rather than being silently
        # relabelled as an overall-timeout.
        done = []
        for task, target in zip(tasks, targets, strict=True):
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is None:
                    done.append(task.result())
                    continue
                # Task finished by raising: preserve the error so it
                # shows up in the summary instead of being masked.
                done.append(
                    Result(
                        target=target,
                        connected=False,
                        attempts=0,
                        error=f"probe raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            # Task did not finish within the wall-clock cap.
            task.cancel()
            done.append(
                Result(
                    target=target,
                    connected=False,
                    attempts=0,
                    error="overall-timeout",
                )
            )
        # Allow cancelled tasks to finalise so we don't leak warnings.
        await asyncio.gather(*tasks, return_exceptions=True)
    return list(done)


def render_summary(
    results: list[Result],
    *,
    mode: Mode,
    connect_timeout: float,
    max_attempts: int,
) -> tuple[str, int]:
    """Render a markdown summary and return (text, failure_count)."""

    # Preserve duplicate entries: keying by label would silently hide
    # misconfigurations where the same host:port is listed twice.
    ordered = sorted(results, key=lambda r: r.target.label)

    if mode == "permitted":
        heading = "## Test PERMITTED Connections"
        blurb = (
            "Endpoints expected to be reachable through the "
            "`CONNECTION_WHITELIST`. Each probe is retried up to "
            f"**{max_attempts}** times with a **{connect_timeout:g}s** "
            "per-attempt timeout."
        )
        columns = "| Endpoint | Expected | Attempts | Result |"
        sep = "| -------- | -------- | -------- | ------ |"
    else:
        heading = "## Test DENIED Connections"
        blurb = (
            "Endpoints expected to be blocked by harden-runner (not "
            "present in `CONNECTION_WHITELIST`). Each probe is a "
            f"single **{connect_timeout:g}s** attempt; any failure to "
            "connect within that attempt is treated as blocked "
            "(harden-runner redirects denied DNS lookups to its "
            "sinkhole address and refuses the TCP connect, so "
            '"Connection refused" is the expected outcome, but '
            "DROP-style firewalls may time out instead)."
        )
        columns = "| Endpoint | Expected | Result |"
        sep = "| -------- | -------- | ------ |"

    lines = [heading, "", blurb, "", columns, sep]
    failures = 0
    for result in ordered:
        if mode == "permitted":
            if result.connected:
                verdict = "connected"
            else:
                verdict = f"FAILED ({result.error or 'no connection'})"
                failures += 1
            lines.append(f"| `{result.target.label}` | allowed | {result.attempts} | {verdict} |")
        else:
            # A probe is only "blocked" if it was actually attempted.
            # An overall-timeout (or any other reason we never produced
            # a real attempt) leaves the endpoint untested; treat that
            # as a failure so a run that hits the wall-clock cap does
            # not silently report 0 leaks.
            incomplete = result.attempts == 0
            if result.connected:
                verdict = "LEAKED (connected)"
                failures += 1
            elif incomplete:
                verdict = f"INCOMPLETE ({result.error or 'not attempted'})"
                failures += 1
            else:
                verdict = f"blocked ({result.error or 'no connection'})"
            lines.append(f"| `{result.target.label}` | blocked | {verdict} |")

    total = len(ordered)
    label = "Failures" if mode == "permitted" else "Leaks"
    lines.extend(["", f"**{label}:** {failures} / {total}", ""])
    return "\n".join(lines), failures


def append_summary(text: str) -> None:
    """Append ``text`` to ``$GITHUB_STEP_SUMMARY`` when available."""

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        print(
            f"::warning::Could not write to GITHUB_STEP_SUMMARY: {exc}",
            file=sys.stderr,
        )


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Install signal handlers so cancellation is near-instant."""

    def _cancel_all() -> None:
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _cancel_all)
        except (NotImplementedError, RuntimeError):
            # Windows / restricted environments: fall back silently.
            pass


def _positive_float(value: str) -> float:
    """Argparse type: parse a strictly positive float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    """Argparse type: parse a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Probe a list of host:port endpoints and assert they are "
            "reachable (permitted mode) or blocked (denied mode)."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("permitted", "denied"),
        help="Test mode: 'permitted' (must connect) or 'denied' (must fail).",
    )
    parser.add_argument(
        "--targets-env",
        required=True,
        help="Name of the environment variable holding the target list.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=_positive_float,
        default=5.0,
        help="Per-attempt TCP connect timeout in seconds, > 0 (default: 5).",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=3,
        help="Max attempts per permitted target, >= 1 (default: 3).",
    )
    parser.add_argument(
        "--overall-timeout",
        type=_positive_float,
        default=60.0,
        help="Hard wall-clock cap for the whole run in seconds, > 0 (default: 60).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""

    args = build_parser().parse_args(argv)
    raw = os.environ.get(args.targets_env, "")
    if not raw.strip():
        print(
            f"::error::Environment variable {args.targets_env} is empty or unset",
            file=sys.stderr,
        )
        return 2
    try:
        targets = parse_targets(raw)
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    if not targets:
        print(
            f"::error::No targets parsed from {args.targets_env}",
            file=sys.stderr,
        )
        return 2

    mode: Mode = args.mode
    print(
        f"Probing {len(targets)} target(s) in {mode} mode "
        f"(connect_timeout={args.connect_timeout}s, "
        f"max_attempts={args.max_attempts}, "
        f"overall_timeout={args.overall_timeout}s)"
    )
    for target in targets:
        print(f"  - {target.label}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)
    start = time.monotonic()
    try:
        results = loop.run_until_complete(
            run_probes(
                targets,
                mode=mode,
                connect_timeout=args.connect_timeout,
                max_attempts=args.max_attempts,
                overall_timeout=args.overall_timeout,
            )
        )
    except asyncio.CancelledError:
        # Signal handlers cancelled every pending task. Give those
        # tasks one quick spin on the loop to finalise — this avoids
        # "Task was destroyed but it is pending!" warnings and closes
        # any half-open transports cleanly. The cancellation is
        # authoritative, so ``return_exceptions=True`` swallows the
        # re-raised CancelledErrors without failing the drain.
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:  # pragma: no cover - defensive only
                pass
        print("::warning::Probe run was cancelled", file=sys.stderr)
        return 130
    finally:
        loop.close()
    elapsed = time.monotonic() - start

    summary, failures = render_summary(
        results,
        mode=mode,
        connect_timeout=args.connect_timeout,
        max_attempts=args.max_attempts,
    )
    summary += f"_Completed in {elapsed:.1f}s._\n"
    append_summary(summary)
    print(summary)

    if failures:
        if mode == "permitted":
            msg = f"{failures} endpoint(s) were unexpectedly blocked"
        else:
            msg = (
                f"{failures} endpoint(s) were unexpectedly reached or "
                "left untested (see summary for per-endpoint verdicts)"
            )
        print(f"::error::{msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
