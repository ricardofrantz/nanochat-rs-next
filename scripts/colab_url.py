#!/usr/bin/env python3
"""
Print a Colab URL for a notebook in this git repo.

Examples:
  python3 scripts/colab_url.py
  python3 scripts/colab_url.py --branch master
  python3 scripts/colab_url.py --open
"""

from __future__ import annotations

import argparse
import re
import subprocess
import webbrowser


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def parse_github_repo(remote_url: str) -> str:
    https_match = re.match(r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?$", remote_url)
    if https_match:
        return https_match.group(1)
    ssh_match = re.match(r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$", remote_url)
    if ssh_match:
        return ssh_match.group(1)
    raise ValueError(
        f"unsupported remote URL: {remote_url} (expected GitHub HTTPS/SSH URL)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Colab URL for a repo notebook.")
    parser.add_argument("--remote", default="origin", help="git remote name")
    parser.add_argument(
        "--branch",
        default="",
        help="branch to use (default: current branch)",
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/colab_gpu_benchmark.ipynb",
        help="path to notebook in repo",
    )
    parser.add_argument("--open", action="store_true", help="open URL in browser")
    args = parser.parse_args()

    remote_url = run("git", "config", "--get", f"remote.{args.remote}.url")
    repo = parse_github_repo(remote_url)
    branch = args.branch or run("git", "rev-parse", "--abbrev-ref", "HEAD")
    url = (
        "https://colab.research.google.com/github/"
        f"{repo}/blob/{branch}/{args.notebook}"
    )
    print(url)
    if args.open:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
