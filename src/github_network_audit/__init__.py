# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""GitHub Network Audit tool."""

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    """Resolve the package version from installed metadata or VCS file."""
    try:
        return version("github-network-audit")
    except PackageNotFoundError:  # pragma: no cover
        # Package is not installed (e.g. running from a source checkout
        # without a build); fall back to the hatch-vcs generated file.
        try:
            from ._version import __version__ as vcs_version
        except ModuleNotFoundError:
            return "0.0.0+unknown"
        return str(vcs_version)


__version__ = _resolve_version()
