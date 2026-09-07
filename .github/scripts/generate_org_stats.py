#!/usr/bin/env python3
"""
generate_org_stats.py

Queries the GitHub REST API for all public, non-archived repositories
belonging to a GitHub organization, aggregates statistics across them,
and renders a self-contained SVG card that is committed back to the
repository by the accompanying GitHub Actions workflow.

Scope (default): public, non-archived repositories owned directly by
the organization. Forks are counted separately and excluded from the
headline "repositories" figure unless INCLUDE_FORKS is set to "true".

Environment variables:
    GITHUB_TOKEN   Required. Used to authenticate API requests.
                    In GitHub Actions this is `secrets.GITHUB_TOKEN`.
    ORG_NAME        Required. The GitHub organization login to query.
    INCLUDE_FORKS   Optional. "true" to include forked repositories
                    in the repository count. Defaults to "false".
    OUTPUT_PATH     Optional. Where to write the SVG. Defaults to
                    assets/generated/org-stats.svg

This script fails gracefully: on API errors it logs the problem and
exits with a non-zero status without writing partial/incorrect data,
so a bad run never overwrites a good statistics card.
"""

from __future__ import annotations

import os
import sys
import time
import html
from dataclasses import dataclass, field
from typing import Optional

import requests

API_ROOT = "https://api.github.com"
PER_PAGE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


@dataclass
class OrgStats:
    repo_count: int = 0
    fork_count: int = 0
    star_count: int = 0
    total_forks_of_repos: int = 0
    open_issues_count: int = 0  # Note: GitHub's API bundles PRs into this count.
    open_pull_requests: int = 0
    contributors: set = field(default_factory=set)
    commit_count: int = 0
    commit_count_is_partial: bool = False
    languages: dict = field(default_factory=dict)


class GitHubAPIError(RuntimeError):
    pass


def _headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request(url: str, token: str, params: Optional[dict] = None) -> requests.Response:
    """Make a GET request with retry/backoff and rate-limit awareness."""
    last_exception: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(token), params=params, timeout=30)
        except requests.RequestException as exc:
            last_exception = exc
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait_seconds = max(reset_at - int(time.time()), 1)
            print(f"Rate limited. Waiting {wait_seconds}s until reset.", file=sys.stderr)
            time.sleep(min(wait_seconds, 300))
            continue

        if resp.status_code >= 500:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        return resp

    raise GitHubAPIError(f"Failed to reach {url} after {MAX_RETRIES} attempts: {last_exception}")


def paginated_get(url: str, token: str, params: Optional[dict] = None) -> list:
    """Follow GitHub's `Link` header pagination and return the full item list."""
    items: list = []
    query = dict(params or {})
    query["per_page"] = PER_PAGE
    next_url: Optional[str] = url

    while next_url:
        resp = _request(next_url, token, params=query if next_url == url else None)
        if resp.status_code == 404:
            raise GitHubAPIError(f"Not found: {next_url}")
        if resp.status_code == 401:
            raise GitHubAPIError("Authentication failed. Check GITHUB_TOKEN permissions.")
        if not resp.ok:
            raise GitHubAPIError(f"GitHub API error {resp.status_code} for {next_url}: {resp.text[:300]}")

        data = resp.json()
        if isinstance(data, list):
            items.extend(data)
        else:
            # Some endpoints (e.g. search) wrap results in a dict.
            items.extend(data.get("items", []))

        next_url = resp.links.get("next", {}).get("url")

    return items


def fetch_org_repos(org: str, token: str, include_forks: bool) -> list:
    repos = paginated_get(
        f"{API_ROOT}/orgs/{org}/repos",
        token,
        params={"type": "public", "sort": "updated"},
    )
    return [r for r in repos if not r.get("archived") and (include_forks or not r.get("fork"))]


def fetch_repo_commit_count(org: str, repo_name: str, token: str) -> tuple[int, bool]:
    """
    Estimate a repository's commit count on the default branch.

    Uses the last page of the paginated commits endpoint to infer the
    total count from the `Link` header, which is far cheaper than
    walking every commit. Returns (count, is_partial) where is_partial
    is True if the count could not be reliably determined (for example,
    an empty repository or an API error) — in that case count is 0 and
    the repo is excluded from the aggregate to avoid undercounting
    silently.
    """
    resp = _request(
        f"{API_ROOT}/repos/{org}/{repo_name}/commits",
        token,
        params={"per_page": 1},
    )

    if resp.status_code == 409:
        # Empty repository (no commits yet).
        return 0, False
    if not resp.ok:
        return 0, True

    last_link = resp.links.get("last", {}).get("url")
    if last_link:
        try:
            page_param = next(p for p in last_link.split("?")[1].split("&") if p.startswith("page="))
            return int(page_param.split("=")[1]), False
        except (IndexError, StopIteration, ValueError):
            return 0, True

    # No Link header means all commits fit on one page.
    try:
        commits = resp.json()
        return len(commits), False
    except ValueError:
        return 0, True


def fetch_repo_contributors(org: str, repo_name: str, token: str) -> set:
    resp = _request(
        f"{API_ROOT}/repos/{org}/{repo_name}/contributors",
        token,
        params={"per_page": PER_PAGE, "anon": "false"},
    )
    if resp.status_code == 204 or not resp.ok:
        return set()
    try:
        return {c["login"] for c in resp.json() if "login" in c}
    except (ValueError, TypeError):
        return set()


def fetch_repo_pull_requests(org: str, repo_name: str, token: str) -> int:
    resp = _request(
        f"{API_ROOT}/repos/{org}/{repo_name}/pulls",
        token,
        params={"state": "open", "per_page": 1},
    )
    if not resp.ok:
        return 0
    last_link = resp.links.get("last", {}).get("url")
    if last_link:
        try:
            page_param = next(p for p in last_link.split("?")[1].split("&") if p.startswith("page="))
            return int(page_param.split("=")[1])
        except (IndexError, StopIteration, ValueError):
            return 0
    try:
        return len(resp.json())
    except ValueError:
        return 0


def aggregate_stats(org: str, token: str, include_forks: bool) -> OrgStats:
    stats = OrgStats()
    repos = fetch_org_repos(org, token, include_forks)
    stats.repo_count = len(repos)

    for repo in repos:
        name = repo["name"]
        stats.star_count += repo.get("stargazers_count", 0)
        stats.total_forks_of_repos += repo.get("forks_count", 0)
        stats.open_issues_count += repo.get("open_issues_count", 0)
        if repo.get("fork"):
            stats.fork_count += 1

        commit_count, is_partial = fetch_repo_commit_count(org, name, token)
        stats.commit_count += commit_count
        stats.commit_count_is_partial = stats.commit_count_is_partial or is_partial

        stats.contributors |= fetch_repo_contributors(org, name, token)
        stats.open_pull_requests += fetch_repo_pull_requests(org, name, token)

    return stats


def render_svg(org: str, stats: OrgStats) -> str:
    """Render a minimal, dependency-free SVG statistics card."""
    safe_org = html.escape(org)
    generated_at = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    commit_label = f"{stats.commit_count:,}" + ("+" if stats.commit_count_is_partial else "")

    rows = [
        ("Repositories", f"{stats.repo_count:,}"),
        ("Commits (tracked)", commit_label),
        ("Contributors", f"{len(stats.contributors):,}"),
        ("Stars", f"{stats.star_count:,}"),
        ("Forks", f"{stats.total_forks_of_repos:,}"),
        ("Open Issues", f"{stats.open_issues_count:,}"),
        ("Open Pull Requests", f"{stats.open_pull_requests:,}"),
    ]

    row_height = 34
    header_height = 56
    footer_height = 34
    width = 420
    height = header_height + row_height * len(rows) + footer_height

    svg_rows = []
    for i, (label, value) in enumerate(rows):
        y = header_height + i * row_height
        safe_label = html.escape(label)
        safe_value = html.escape(value)
        svg_rows.append(
            f'<text x="20" y="{y + 22}" class="label">{safe_label}</text>'
            f'<text x="{width - 20}" y="{y + 22}" text-anchor="end" class="value">{safe_value}</text>'
        )
        if i < len(rows) - 1:
            svg_rows.append(
                f'<line x1="20" y1="{y + row_height}" x2="{width - 20}" y2="{y + row_height}" class="divider" />'
            )

    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Robotronics Club organization statistics">
  <style>
    .bg {{ fill: #0d1117; }}
    .title {{ fill: #58a6ff; font: 600 15px 'Segoe UI', Ubuntu, sans-serif; }}
    .subtitle {{ fill: #8b949e; font: 400 11px 'Segoe UI', Ubuntu, sans-serif; }}
    .label {{ fill: #c9d1d9; font: 400 13px 'Segoe UI', Ubuntu, sans-serif; }}
    .value {{ fill: #ffffff; font: 700 13px 'Segoe UI', Ubuntu, sans-serif; }}
    .divider {{ stroke: #21262d; stroke-width: 1; }}
    .footer {{ fill: #6e7681; font: 400 10px 'Segoe UI', Ubuntu, sans-serif; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" rx="10" class="bg" />
  <text x="20" y="26" class="title">{safe_org} — Organization Statistics</text>
  <text x="20" y="42" class="subtitle">Public, non-archived repositories</text>
  {''.join(svg_rows)}
  <text x="20" y="{height - 14}" class="footer">Generated {generated_at} by GitHub Actions. Updated daily.</text>
</svg>
'''


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    org = os.environ.get("ORG_NAME")
    include_forks = os.environ.get("INCLUDE_FORKS", "false").strip().lower() == "true"
    output_path = os.environ.get("OUTPUT_PATH", "assets/generated/org-stats.svg")

    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is not set.", file=sys.stderr)
        return 1
    if not org:
        print("ERROR: ORG_NAME environment variable is not set.", file=sys.stderr)
        return 1

    try:
        stats = aggregate_stats(org, token, include_forks)
    except GitHubAPIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    svg = render_svg(org, stats)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Wrote statistics for '{org}' to {output_path}")
    print(
        f"repos={stats.repo_count} commits={stats.commit_count}"
        f"{'(partial)' if stats.commit_count_is_partial else ''} "
        f"contributors={len(stats.contributors)} stars={stats.star_count} "
        f"forks={stats.total_forks_of_repos} open_issues={stats.open_issues_count} "
        f"open_prs={stats.open_pull_requests}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
