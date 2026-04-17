# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Data collection from GitHub GraphQL and StepSecurity APIs."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

STEPSECURITY_API = "https://agent.api.stepsecurity.io/v1"
GITHUB_GRAPHQL_API = "https://api.github.com/graphql"


class NetworkAuditCollector:
    """Collect network endpoint data from GitHub org workflow runs."""

    def __init__(
        self,
        org: str,
        output_dir: Path,
        github_token: str | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            org: GitHub organization name.
            output_dir: Base directory for cached data.
            github_token: GitHub API token for GraphQL queries.
        """
        self.org = org
        self.base_dir = output_dir / org
        self.github_token = github_token
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _cache_path(self, *parts: str) -> Path:
        """Build a cache file path.

        Args:
            *parts: Path components relative to base_dir.

        Returns:
            Full path to the cache file.
        """
        return self.base_dir.joinpath(*parts)

    def _read_cache(self, *parts: str) -> dict | list | None:
        """Read cached JSON data from disk.

        Args:
            *parts: Path components relative to base_dir.

        Returns:
            Parsed JSON data or None if not cached or corrupt.
        """
        path = self._cache_path(*parts)
        if path.exists():
            try:
                result: dict | list = json.loads(
                    path.read_text(encoding="utf-8"),
                )
            except json.JSONDecodeError:
                logger.warning("Corrupt cache file: %s", path)
                return None
            return result
        return None

    def _write_cache(self, data: dict | list, *parts: str) -> Path:
        """Write data to cache as JSON.

        Args:
            data: Data to cache.
            *parts: Path components relative to base_dir.

        Returns:
            Path to the written cache file.
        """
        path = self._cache_path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def fetch_repos(self, *, refresh: bool = False) -> list[dict]:
        """Fetch all repositories in the org via GitHub GraphQL.

        Args:
            refresh: Force refresh even if cached data exists.

        Returns:
            List of repository metadata dicts.
        """
        if not refresh:
            cached = self._read_cache("repos.json")
            if cached is not None:
                logger.info("Using cached repos (%d repos)", len(cached))
                return cached  # type: ignore[return-value]

        if not self.github_token:
            msg = "GITHUB_TOKEN required for GraphQL API"
            raise ValueError(msg)

        query = """
        query($org: String!, $cursor: String) {
            organization(login: $org) {
                repositories(first: 100, after: $cursor) {
                    totalCount
                    pageInfo { hasNextPage endCursor }
                    nodes {
                        name
                        isArchived
                        isPrivate
                        defaultBranchRef { name }
                    }
                }
            }
        }
        """
        all_repos: list[dict] = []
        cursor = None

        while True:
            variables: dict = {"org": self.org, "cursor": cursor}
            resp = self.session.post(
                GITHUB_GRAPHQL_API,
                json={"query": query, "variables": variables},
                headers={"Authorization": f"bearer {self.github_token}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                msg = f"GraphQL error: {data['errors']}"
                raise RuntimeError(msg)

            repos_data = data["data"]["organization"]["repositories"]
            all_repos.extend(repos_data["nodes"])

            if not repos_data["pageInfo"]["hasNextPage"]:
                break
            cursor = repos_data["pageInfo"]["endCursor"]

        logger.info("Fetched %d repositories from %s", len(all_repos), self.org)
        self._write_cache(all_repos, "repos.json")
        return all_repos

    def fetch_repo_runs(
        self,
        repo: str,
        *,
        refresh: bool = False,
    ) -> list[dict]:
        """Fetch all workflow runs for a repo from StepSecurity.

        Args:
            repo: Repository name.
            refresh: Force refresh even if cached data exists.

        Returns:
            List of workflow run summary dicts.
        """
        if not refresh:
            cached = self._read_cache(repo, "runs.json")
            if cached is not None:
                logger.info(
                    "Using cached runs for %s (%d runs)",
                    repo,
                    len(cached),
                )
                return cached  # type: ignore[return-value]

        seen_ids: set[str] = set()
        all_runs: list[dict] = []
        page = 1

        while True:
            url = f"{STEPSECURITY_API}/github/{self.org}/{repo}/actions/runs?page={page}"
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException:
                logger.warning(
                    "Request failed for %s page %d",
                    repo,
                    page,
                )
                break

            if resp.status_code == 404:
                logger.info(
                    "No StepSecurity data for %s/%s",
                    self.org,
                    repo,
                )
                break

            if resp.status_code != 200:
                logger.warning(
                    "HTTP %d for %s/%s page %d",
                    resp.status_code,
                    self.org,
                    repo,
                    page,
                )
                break

            data = resp.json()
            runs = data.get("workflow_runs", [])

            new_runs = [r for r in runs if r["id"] not in seen_ids]
            if not new_runs:
                logger.debug(
                    "%s: page %d has only duplicates",
                    repo,
                    page,
                )
                break

            for r in new_runs:
                seen_ids.add(r["id"])
            all_runs.extend(new_runs)

            total_pages = data.get("total_pages", 1)
            logger.debug(
                "%s: page %d/%d (%d new, %d total)",
                repo,
                page,
                total_pages,
                len(new_runs),
                len(all_runs),
            )

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.3)

        logger.info(
            "Fetched %d unique runs for %s/%s",
            len(all_runs),
            self.org,
            repo,
        )
        self._write_cache(all_runs, repo, "runs.json")
        return all_runs

    def fetch_run_detail(
        self,
        repo: str,
        run_id: str,
        *,
        refresh: bool = False,
    ) -> dict | None:
        """Fetch detailed run data from StepSecurity.

        Args:
            repo: Repository name.
            run_id: Workflow run ID.
            refresh: Force refresh even if cached data exists.

        Returns:
            Run detail dict or None on failure.
        """
        cache_parts = (repo, "runs", f"{run_id}.json")

        if not refresh:
            cached = self._read_cache(*cache_parts)
            if cached is not None:
                return cached  # type: ignore[return-value]

        url = f"{STEPSECURITY_API}/github/{self.org}/{repo}/actions/runs/{run_id}"
        try:
            resp = self.session.get(url, timeout=30)
        except requests.RequestException:
            logger.warning(
                "Request failed for run %s in %s",
                run_id,
                repo,
            )
            return None

        if resp.status_code != 200:
            logger.warning(
                "HTTP %d for run %s in %s",
                resp.status_code,
                run_id,
                repo,
            )
            return None

        data: dict = resp.json()
        self._write_cache(data, *cache_parts)
        return data

    def fetch_run_details_batch(
        self,
        repo: str,
        run_ids: list[str],
        *,
        refresh: bool = False,
        workers: int = 8,
    ) -> dict[str, dict | None]:
        """Fetch multiple run details in parallel.

        Args:
            repo: Repository name.
            run_ids: List of workflow run IDs.
            refresh: Force refresh even if cached data exists.
            workers: Maximum concurrent requests.

        Returns:
            Mapping of run_id to detail dict (or None).
        """
        results: dict[str, dict | None] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self.fetch_run_detail,
                    repo,
                    rid,
                    refresh=refresh,
                ): rid
                for rid in run_ids
            }
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    results[rid] = future.result()
                except Exception:
                    logger.warning(
                        "Unexpected error for run %s in %s",
                        rid,
                        repo,
                    )
                    results[rid] = None

        return results
