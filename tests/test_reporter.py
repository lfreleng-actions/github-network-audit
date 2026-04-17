# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the network audit reporter module."""

from __future__ import annotations

from pathlib import Path

import pytest

from github_network_audit.reporter import (
    NetworkAuditReporter,
    extract_endpoints_from_run,
)

SAMPLE_RUN_DETAIL = {
    "id": "12345",
    "repo": "lfreleng-actions/test-repo",
    "path": ".github/workflows/test.yaml",
    "jobs": [
        {
            "name": "Build",
            "harden_runner_egress_policy": "audit",
            "steps": [
                {
                    "name": "Checkout",
                    "action": "actions/checkout",
                    "tools": [
                        {
                            "name": "git-remote-http",
                            "endpoints": [
                                {
                                    "domainName": "github.com",
                                    "port": "443",
                                    "friendlyName": "GitHub",
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "Install deps",
                    "action": "",
                    "tools": [
                        {
                            "name": "uv",
                            "endpoints": [
                                {
                                    "domainName": "pypi.org",
                                    "port": "443",
                                    "friendlyName": "Python Registry",
                                },
                                {
                                    "domainName": "files.pythonhosted.org",
                                    "port": "443",
                                    "friendlyName": "Python Registry",
                                },
                            ],
                        },
                    ],
                },
                {
                    "name": "No network",
                    "action": "",
                    "tools": [],
                },
            ],
        },
    ],
}


class TestExtractEndpoints:
    """Test endpoint extraction from run details."""

    def test_extracts_all_endpoints(self) -> None:
        """Verify all endpoints from all tools get extracted."""
        endpoints = extract_endpoints_from_run(SAMPLE_RUN_DETAIL)
        assert len(endpoints) == 3

    def test_endpoint_fields(self) -> None:
        """Verify extracted endpoint fields match source data."""
        endpoints = extract_endpoints_from_run(SAMPLE_RUN_DETAIL)
        github_ep = next(e for e in endpoints if e["domain"] == "github.com")
        assert github_ep["port"] == "443"
        assert github_ep["process"] == "git-remote-http"
        assert github_ep["friendly_name"] == "GitHub"
        assert github_ep["repo"] == "lfreleng-actions/test-repo"
        assert github_ep["workflow"] == ".github/workflows/test.yaml"
        assert github_ep["run_id"] == "12345"

    def test_empty_run(self) -> None:
        """Verify empty run detail produces no endpoints."""
        endpoints = extract_endpoints_from_run({})
        assert endpoints == []

    def test_step_without_tools(self) -> None:
        """Verify steps with no tools produce no endpoints."""
        run: dict = {
            "jobs": [{"steps": [{"tools": []}]}],
        }
        endpoints = extract_endpoints_from_run(run)
        assert endpoints == []


class TestBuildAllowlist:
    """Test allowlist deduplication logic."""

    def test_deduplicates_by_domain_port(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify endpoints deduplicate on domain:port key."""
        reporter = NetworkAuditReporter(
            org="test-org",
            output_dir=tmp_path,
        )
        endpoints = [
            {
                "domain": "github.com",
                "port": "443",
                "friendly_name": "GitHub",
                "process": "git",
                "repo": "org/repo-a",
                "workflow": "a.yaml",
                "action": "actions/checkout",
            },
            {
                "domain": "github.com",
                "port": "443",
                "friendly_name": "GitHub",
                "process": "node",
                "repo": "org/repo-b",
                "workflow": "b.yaml",
                "action": "actions/setup-node",
            },
        ]
        allowlist = reporter._build_allowlist(endpoints)
        assert len(allowlist) == 1
        entry = allowlist[0]
        assert entry["endpoint"] == "github.com:443"
        assert set(entry["processes"]) == {"git", "node"}
        assert set(entry["repos"]) == {"org/repo-a", "org/repo-b"}
        assert entry["repo_count"] == 2

    def test_sorts_by_domain(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify allowlist output sorts alphabetically."""
        reporter = NetworkAuditReporter(
            org="test-org",
            output_dir=tmp_path,
        )
        endpoints = [
            {
                "domain": "pypi.org",
                "port": "443",
                "friendly_name": "",
                "process": "uv",
                "repo": "org/r",
                "workflow": "w",
                "action": "",
            },
            {
                "domain": "github.com",
                "port": "443",
                "friendly_name": "",
                "process": "git",
                "repo": "org/r",
                "workflow": "w",
                "action": "",
            },
        ]
        allowlist = reporter._build_allowlist(endpoints)
        assert allowlist[0]["domain"] == "github.com"
        assert allowlist[1]["domain"] == "pypi.org"


class TestGenerateReports:
    """Test report generation."""

    def test_invalid_format_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify unknown format raises ValueError."""
        reporter = NetworkAuditReporter(
            org="test-org",
            output_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="Unknown format"):
            reporter.generate_reports(output_format="xml")
