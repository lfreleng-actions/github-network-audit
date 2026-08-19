# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Deployment configuration for external service endpoints.

Each endpoint reads from the environment first so a mirror or test
double can override it, falling back to the public production URL.
"""

from __future__ import annotations

import os

STEPSECURITY_API: str = os.environ.get(
    "STEPSECURITY_API_URL",
    # aislop-ignore-next-line ai-slop/hardcoded-url -- stable public API base; env override above
    "https://agent.api.stepsecurity.io/v1",
)
GITHUB_GRAPHQL_API: str = os.environ.get(
    "GITHUB_GRAPHQL_URL",
    "https://api.github.com/graphql",
)
