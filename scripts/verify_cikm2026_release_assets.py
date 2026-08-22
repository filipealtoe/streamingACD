#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Verify the CIKM 2026 GitHub release assets from public metadata only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# CAMERA-READY ARTIFACT CHANGE | Author: Sérgio Pinto | Timestamp:
# 2026-08-21 22:29 PDT | Reason: provide a deterministic, read-only check of
# the externally hosted checkpoint and embedding identities without downloading
# either large binary.

DEFAULT_MANIFEST = Path("reproducibility/cikm2026/RELEASE_ASSETS.json")
GITHUB_API_VERSION = "2022-11-28"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def fetch_release(repository: str, tag: str, timeout: float) -> dict[str, Any]:
    encoded_tag = urllib.parse.quote(tag, safe="")
    url = f"https://api.github.com/repos/{repository}/releases/tags/{encoded_tag}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamingACD-CIKM2026-release-verifier",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("GitHub returned a non-object release response")
    return data


def compare_release(
    manifest: dict[str, Any], release: dict[str, Any]
) -> list[tuple[str, bool, str]]:
    expected_release = manifest["release"]
    checks: list[tuple[str, bool, str]] = []
    release_fields = {
        "tag": release.get("tag_name"),
        "draft": release.get("draft"),
        "prerelease": release.get("prerelease"),
    }
    for field, observed in release_fields.items():
        expected = expected_release[field]
        checks.append(
            (
                f"Release {field}",
                observed == expected,
                f"expected={expected!r}; observed={observed!r}",
            )
        )

    observed_assets = {
        asset.get("name"): asset
        for asset in release.get("assets", [])
        if isinstance(asset, dict) and isinstance(asset.get("name"), str)
    }
    for expected_asset in manifest["assets"]:
        if not expected_asset.get("required", False):
            continue
        name = expected_asset["name"]
        observed = observed_assets.get(name)
        if observed is None:
            checks.append((f"Asset {name}", False, "missing from public release"))
            continue
        expected_size = expected_asset["bytes"]
        expected_digest = expected_asset["digest"]
        observed_size = observed.get("size")
        observed_digest = observed.get("digest")
        observed_state = observed.get("state")
        passed = (
            observed_size == expected_size
            and observed_digest == expected_digest
            and observed_state == "uploaded"
        )
        checks.append(
            (
                f"Asset {name}",
                passed,
                f"state={observed_state!r}; bytes={observed_size!r}; "
                f"digest={observed_digest!r}",
            )
        )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Expected release-asset manifest relative to the repository root",
    )
    parser.add_argument(
        "--release-json",
        type=Path,
        help="Use a saved GitHub API response instead of making a network request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="GitHub API timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    try:
        manifest = load_json(manifest_path)
        if args.release_json:
            release_path = args.release_json
            if not release_path.is_absolute():
                release_path = repo_root / release_path
            release = load_json(release_path)
            source = release_path.as_posix()
        else:
            release = fetch_release(
                manifest["repository"], manifest["release"]["tag"], args.timeout
            )
            source = "GitHub public release API"
        checks = compare_release(manifest, release)
    except (KeyError, OSError, ValueError, urllib.error.URLError) as exc:
        print(f"[FAIL] Release metadata: {exc}")
        return 1

    print("CIKM 2026 external release-asset verification")
    print(f"Source: {source}")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
    passed_count = sum(passed for _, passed, _ in checks)
    print(f"Summary: {passed_count}/{len(checks)} checks passed")
    return 0 if passed_count == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
