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

## VEXXHOST Infrastructure

The Linux Foundation uses [VEXXHOST](https://vexxhost.com/) for some
infrastructure needs, running servers and software in their data centres.
The `resources/VEXXHOST.txt` file catalogues the IPv4 and IPv6 CIDR
blocks that AS33028 announces, for reference.

**harden-runner does not accept CIDR blocks in its allowlist.** The
`allowed-endpoints` input expects DNS names (`host:port`) rather than
raw IP ranges, so the testing workflow in this repository ignores the
file and you MUST NOT add the CIDR entries to `CONNECTION_WHITELIST`.
When a workflow reports blocked egress to an unfamiliar IP, resolve
the destination manually (for example with `whois` or
[bgp.he.net](https://bgp.he.net/AS33028)) and compare against the CIDR
list in the resource file. If the address falls inside an AS33028 range,
adding the corresponding LF-operated hostname (for example
`gerrit.linuxfoundation.org:443`) to the allowlist is the correct fix.
CIDR-based egress enforcement requires mechanisms outside
harden-runner, such as iptables on self-hosted runners or the
StepSecurity Policy Store.

Because provider allocations change over time, verify or refresh the
CIDR list against a current authoritative view of AS33028 when
troubleshooting; the resource file records its source URL at the top.
