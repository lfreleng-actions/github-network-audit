<!--
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation
-->

# GitHub Network Audit

Collect outbound network connection data from GitHub Actions workflow
runs that use
[step-security/harden-runner](https://github.com/step-security/harden-runner).

This tool gathers network endpoint data from the StepSecurity API and
builds consolidated allowlists for use with harden-runner's
`egress-policy: block` mode.

## Installation

```bash
uv tool install -e .
```

## Usage

### Collect Data

<!-- markdownlint-disable MD013 -->

```bash
# Set your GitHub token (needed for repo enumeration via GraphQL)
export GITHUB_TOKEN="your-token"

# Collect data for the entire org
github-network-audit collect --org lfreleng-actions

# Collect data for a specific repo
github-network-audit collect --org lfreleng-actions --repo path-check-action

# Force refresh of cached data
github-network-audit collect --org lfreleng-actions --refresh
```

### Generate Reports

```bash
# All formats (JSON, CSV, Markdown)
github-network-audit report --org lfreleng-actions

# Specific format
github-network-audit report --org lfreleng-actions --output-format md

# Report for specific repo
github-network-audit report --org lfreleng-actions --repo path-check-action
```

<!-- markdownlint-enable MD013 -->

## Output

The tool writes reports to the `{org}/` directory:

- `all_endpoints.json` - Complete endpoint inventory
- `allowlist.json` - Deduplicated allowlist with metadata
- `allowlist.csv` - Spreadsheet-friendly format
- `allowlist.md` - Markdown with harden-runner config snippet

Use `--repo` to produce repo-scoped output files
(e.g. `allowlist_path-check-action.json`).

## Data Sources

- **GitHub GraphQL API** - Repository enumeration (single query)
- **StepSecurity API** - Network endpoint data (unauthenticated)

The tool caches all data locally for idempotent operation.

## Notes

All intermediate data persists on disk. Follow-up runs use cached data
by default. Use `--refresh` to force re-fetching from APIs.
