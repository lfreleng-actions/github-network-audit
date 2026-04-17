# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Report generation from cached network audit data."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_endpoints_from_run(run_detail: dict) -> list[dict]:
    """Extract network endpoints from a single run detail.

    Args:
        run_detail: Full run detail from StepSecurity API.

    Returns:
        List of endpoint record dicts.
    """
    endpoints: list[dict] = []
    repo = run_detail.get("repo", "")
    workflow = run_detail.get("path", "")
    run_id = str(run_detail.get("id", ""))

    for job in run_detail.get("jobs", []):
        job_name = job.get("name", "")
        egress_policy = job.get("harden_runner_egress_policy", "")

        for step in job.get("steps", []):
            step_name = step.get("name", "")
            action = step.get("action", "")

            for tool in step.get("tools", []):
                process = tool.get("name", "")

                for ep in tool.get("endpoints", []):
                    endpoints.append(
                        {
                            "domain": ep.get("domainName", ""),
                            "port": ep.get("port", ""),
                            "friendly_name": ep.get("friendlyName", ""),
                            "process": process,
                            "step": step_name,
                            "action": action,
                            "job": job_name,
                            "repo": repo,
                            "workflow": workflow,
                            "run_id": run_id,
                            "egress_policy": egress_policy,
                        }
                    )

    return endpoints


class NetworkAuditReporter:
    """Generate reports from cached network audit data."""

    def __init__(self, org: str, output_dir: Path) -> None:
        """Initialize the reporter.

        Args:
            org: GitHub organization name.
            output_dir: Base directory containing cached data.
        """
        self.org = org
        self.base_dir = output_dir / org

    def _collect_all_endpoints(
        self,
        repo_filter: str | None = None,
    ) -> list[dict]:
        """Extract endpoints from all cached run detail files.

        Args:
            repo_filter: Limit to a specific repository.

        Returns:
            List of all endpoint records.
        """
        endpoints: list[dict] = []

        if not self.base_dir.exists():
            logger.warning("No cached data at %s", self.base_dir)
            return endpoints

        repo_dirs = sorted(self.base_dir.iterdir())
        for repo_dir in repo_dirs:
            if not repo_dir.is_dir():
                continue
            if repo_filter and repo_dir.name != repo_filter:
                continue

            runs_dir = repo_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_file in sorted(runs_dir.glob("*.json")):
                try:
                    run_data = json.loads(
                        run_file.read_text(encoding="utf-8"),
                    )
                    endpoints.extend(
                        extract_endpoints_from_run(run_data),
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "Error parsing %s: %s",
                        run_file,
                        exc,
                    )

        return endpoints

    def _build_allowlist(
        self,
        endpoints: list[dict],
    ) -> list[dict]:
        """Build a deduplicated allowlist from raw endpoints.

        Args:
            endpoints: List of raw endpoint records.

        Returns:
            Sorted, deduplicated allowlist entries.
        """
        seen: dict[tuple[str, str], dict] = {}

        for ep in endpoints:
            key = (ep["domain"], ep["port"])
            if key not in seen:
                seen[key] = {
                    "domain": ep["domain"],
                    "port": ep["port"],
                    "friendly_name": ep.get("friendly_name", ""),
                    "processes": set(),
                    "repos": set(),
                    "workflows": set(),
                    "actions": set(),
                }

            entry = seen[key]
            if ep.get("process"):
                entry["processes"].add(ep["process"])
            if ep.get("repo"):
                entry["repos"].add(ep["repo"])
            if ep.get("workflow"):
                entry["workflows"].add(ep["workflow"])
            if ep.get("action"):
                entry["actions"].add(ep["action"])
            if ep.get("friendly_name") and not entry["friendly_name"]:
                entry["friendly_name"] = ep["friendly_name"]

        result = []
        for entry in sorted(
            seen.values(),
            key=lambda x: x["domain"],
        ):
            result.append(
                {
                    "domain": entry["domain"],
                    "port": entry["port"],
                    "endpoint": f"{entry['domain']}:{entry['port']}",
                    "friendly_name": entry["friendly_name"],
                    "processes": sorted(entry["processes"]),
                    "repos": sorted(entry["repos"]),
                    "repo_count": len(entry["repos"]),
                    "workflows": sorted(entry["workflows"]),
                    "actions": sorted(entry["actions"]),
                }
            )

        return result

    def _write_json(self, data: list[dict], path: Path) -> None:
        """Write data as JSON.

        Args:
            data: Data to write.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote JSON report: %s", path)

    def _write_csv(self, allowlist: list[dict], path: Path) -> None:
        """Write allowlist as CSV.

        Args:
            allowlist: Deduplicated endpoint entries.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        output = io.StringIO()
        fieldnames = [
            "endpoint",
            "domain",
            "port",
            "friendly_name",
            "processes",
            "repo_count",
            "repos",
            "workflows",
            "actions",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in allowlist:
            row = {
                **entry,
                "processes": "; ".join(entry["processes"]),
                "repos": "; ".join(entry["repos"]),
                "workflows": "; ".join(entry["workflows"]),
                "actions": "; ".join(entry["actions"]),
            }
            writer.writerow(row)

        path.write_text(output.getvalue(), encoding="utf-8")
        logger.info("Wrote CSV report: %s", path)

    def _write_markdown(
        self,
        allowlist: list[dict],
        path: Path,
    ) -> None:
        """Write allowlist as Markdown with harden-runner config.

        Args:
            allowlist: Deduplicated endpoint entries.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "<!--",
            # REUSE-IgnoreStart
            "# SPDX-License-Identifier: Apache-2.0",
            "# SPDX-FileCopyrightText: 2026 The Linux Foundation",
            "-->",
            "",
            # REUSE-IgnoreEnd
            f"# Network Allowlist: {self.org}",
            "",
            f"Total unique endpoints: **{len(allowlist)}**",
            "",
            "## Harden-Runner Configuration",
            "",
            "Copy this block into your workflow YAML:",
            "",
            "<!-- markdownlint-disable MD046 -->",
            "",
            "```yaml",
            "- uses: step-security/harden-runner@v2",
            "  with:",
            "    egress-policy: block",
            "    allowed-endpoints: >",
        ]

        for entry in allowlist:
            lines.append(f"      {entry['endpoint']}")
        lines.extend(["```", ""])
        lines.append("<!-- markdownlint-enable MD046 -->")
        lines.append("")

        lines.append("## Endpoint Details")
        lines.append("")
        lines.append("| Endpoint | Name | Processes | Repos |")
        lines.append("| -------- | ---- | --------- | ----- |")
        for entry in allowlist:
            procs = ", ".join(entry["processes"])
            lines.append(f"| `{entry['endpoint']}` | {entry['friendly_name']} | {procs} | {entry['repo_count']} |")
        lines.append("")

        lines.append("## Per-Repository Breakdown")
        lines.append("")
        repo_endpoints: dict[str, list[dict]] = {}
        for entry in allowlist:
            for repo in entry["repos"]:
                if repo not in repo_endpoints:
                    repo_endpoints[repo] = []
                repo_endpoints[repo].append(entry)

        for repo_name in sorted(repo_endpoints):
            short = repo_name.split("/")[-1] if "/" in repo_name else repo_name
            lines.append(f"### {short}")
            lines.append("")
            for entry in repo_endpoints[repo_name]:
                procs = ", ".join(entry["processes"])
                lines.append(f"- `{entry['endpoint']}` ({procs})")
            lines.append("")

        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote Markdown report: %s", path)

    def generate_reports(
        self,
        output_format: str = "all",
        repo_filter: str | None = None,
    ) -> list[dict]:
        """Generate reports from cached data.

        Args:
            output_format: One of 'json', 'csv', 'md', or 'all'.
            repo_filter: Limit to a specific repository name.

        Returns:
            The deduplicated allowlist.

        Raises:
            ValueError: If output_format is not recognized.
        """
        valid_formats = {"json", "csv", "md", "all"}
        if output_format not in valid_formats:
            msg = f"Unknown format '{output_format}'. Valid options: {', '.join(sorted(valid_formats))}"
            raise ValueError(msg)

        all_endpoints = self._collect_all_endpoints(
            repo_filter=repo_filter,
        )
        logger.info(
            "Extracted %d endpoint records",
            len(all_endpoints),
        )

        allowlist = self._build_allowlist(all_endpoints)
        logger.info(
            "Built allowlist with %d unique endpoints",
            len(allowlist),
        )

        suffix = f"_{repo_filter}" if repo_filter else ""

        if output_format in ("json", "all"):
            self._write_json(
                all_endpoints,
                self.base_dir / f"all_endpoints{suffix}.json",
            )
            self._write_json(
                allowlist,
                self.base_dir / f"allowlist{suffix}.json",
            )

        if output_format in ("csv", "all"):
            self._write_csv(
                allowlist,
                self.base_dir / f"allowlist{suffix}.csv",
            )

        if output_format in ("md", "all"):
            self._write_markdown(
                allowlist,
                self.base_dir / f"allowlist{suffix}.md",
            )

        return allowlist
