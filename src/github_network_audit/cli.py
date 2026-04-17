# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""CLI interface for GitHub Network Audit."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import typer
from rich.console import Console

from github_network_audit.collector import NetworkAuditCollector
from github_network_audit.reporter import NetworkAuditReporter

app = typer.Typer(
    name="github-network-audit",
    help="Audit outbound network connections from GitHub Actions.",
)
console = Console()


@app.command()
def collect(
    org: str = typer.Option(
        "lfreleng-actions",
        help="GitHub organization to audit.",
    ),
    repo: str | None = typer.Option(
        None,
        help="Specific repository to audit.",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        help="Output directory for cached data.",
    ),
    refresh: bool = typer.Option(
        False,
        help="Force refresh of cached data.",
    ),
    github_token: str | None = typer.Option(
        None,
        envvar="GITHUB_TOKEN",
        help="GitHub API token for GraphQL queries.",
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        "-w",
        help="Concurrent requests for run details.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging output.",
    ),
) -> None:
    """Collect network endpoint data from workflow runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    collector = NetworkAuditCollector(
        org=org,
        output_dir=output_dir,
        github_token=github_token,
    )

    console.print(
        f"[bold]Fetching repositories for: {org}[/bold]",
    )
    repos = collector.fetch_repos(refresh=refresh)
    active = [r for r in repos if not r.get("isArchived")]
    console.print(
        f"  Found {len(repos)} repos ({len(active)} active)",
    )

    if repo:
        repo_names = [repo]
    else:
        repo_names = [r["name"] for r in active]

    total = len(repo_names)
    for idx, repo_name in enumerate(repo_names, start=1):
        console.print()
        console.print(
            f"[bold cyan]Processing {idx}/{total}: {repo_name}[/bold cyan]",
        )

        runs = collector.fetch_repo_runs(
            repo_name,
            refresh=refresh,
        )
        console.print(f"  Total runs tracked: {len(runs)}")

        detail_runs = [r for r in runs if r.get("destination_count", 0) > 0]
        console.print(
            f"  Runs with network data: {len(detail_runs)}",
        )

        if detail_runs:
            run_ids = [r["id"] for r in detail_runs]
            collector.fetch_run_details_batch(
                repo_name,
                run_ids,
                refresh=refresh,
                workers=workers,
            )
            console.print(
                f"    Fetched {len(detail_runs)} details",
                f"({workers} workers)",
            )

        console.print(
            f"  [green]✓[/green] Done with {repo_name}",
        )

    console.print()
    console.print(
        "[bold green]✓ Collection complete![/bold green]",
    )


@app.command()
def report(
    org: str = typer.Option(
        "lfreleng-actions",
        help="GitHub organization.",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        help="Directory with cached data.",
    ),
    output_format: str = typer.Option(
        "all",
        help="Output format: md, csv, json, or all.",
        click_type=click.Choice(
            ["all", "json", "csv", "md"],
            case_sensitive=False,
        ),
    ),
    repo: str | None = typer.Option(
        None,
        help="Report for specific repo only.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging output.",
    ),
) -> None:
    """Generate reports from cached data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    reporter = NetworkAuditReporter(
        org=org,
        output_dir=output_dir,
    )

    console.print(
        f"[bold]Generating reports for: {org}[/bold]",
    )
    allowlist = reporter.generate_reports(
        output_format=output_format,
        repo_filter=repo,
    )
    console.print(
        f"  Unique endpoints: [bold]{len(allowlist)}[/bold]",
    )

    report_dir = output_dir / org
    console.print(f"  Reports written to: {report_dir}")
    console.print()
    console.print(
        "[bold green]✓ Reports generated![/bold green]",
    )
