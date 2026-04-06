#!/usr/bin/env python3
"""Bump version across pyproject.toml, Cargo.toml, and docs/conf.py."""

import os
import re
import sys

def update_file(path, pattern, replacement):
    with open(path, "r") as f:
        content = f.read()
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"  WARNING: no match in {path}")
        return False
    with open(path, "w") as f:
        f.write(new_content)
    print(f"  {path}: updated")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python update_version.py X.Y.Z")
        sys.exit(1)

    version = sys.argv[1]
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        print(f"Invalid version: {version} (expected X.Y.Z)")
        sys.exit(1)

    print(f"Updating version to {version}...")

    update_file(
        "pyproject.toml",
        r'version\s*=\s*"[^"]*"',
        f'version = "{version}"',
    )
    update_file(
        "Cargo.toml",
        r'version\s*=\s*"[^"]*"',
        f'version = "{version}"',
    )
    update_file(
        "docs/conf.py",
        r"release\s*=\s*'[^']*'",
        f"release = '{version}'",
    )
    if os.path.exists("setup.py"):
        update_file(
            "setup.py",
            r"version\s*=\s*'[^']*'",
            f"version='{version}'",
        )
    else:
        print("  setup.py: skipped (not found)")

    print(f"\nDone. Now run:")
    print(f"  git add -A")
    print(f"  git commit -m 'release: v{version}'")
    print(f"  git tag v{version}")
    print(f"  git push && git push --tags")

if __name__ == "__main__":
    main()
