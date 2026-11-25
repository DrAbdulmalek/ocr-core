#!/usr/bin/env python3
"""Contract guard (issue #73): any PR touching the contract surface must
bump the package MAJOR version.

The contract surface is everything that defines what producers emit and
consumers read: the JSON schema (``src/ocr_core/schemas/``), the in-code
dataclass (``src/ocr_core/engines/base.py``), the contract document
(``docs/CONTRACT.md``) and the golden signature files under ``tests/``.

Exit codes: 0 = gate passes (no contract files touched, or major bumped);
1 = gate failure with remediation instructions.

Stdlib only — runs in CI without installing the package.
Usage:  python scripts/ci/contract_guard.py --base <base git sha>
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

CONTRACT_PATHS = (
    "src/ocr_core/schemas/",
    "src/ocr_core/engines/base.py",
    "docs/CONTRACT.md",
    "tests/test_contract.py",
    "tests/golden_contract_signature.json",
)

VERSION_RE = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return out.stdout


def changed_files(base: str) -> set[str]:
    return set(git("diff", "--name-only", f"{base}..HEAD").split())


def version_at(ref: str) -> tuple[int, int, int]:
    text = git("show", f"{ref}:pyproject.toml")
    m = VERSION_RE.search(text)
    if not m:
        sys.exit(f"✖ could not parse version from pyproject.toml at {ref}")
    return tuple(int(x) for x in m.groups())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base git sha (PR base)")
    args = ap.parse_args()

    files = changed_files(args.base)
    touched = sorted(
        f for f in files
        if any(f == p or f.startswith(p) for p in CONTRACT_PATHS))
    if not touched:
        print("✓ contract guard: no contract files touched — gate not applicable")
        return 0

    base_v = version_at(args.base)
    head_v = version_at("HEAD")
    base_s, head_s = ".".join(map(str, base_v)), ".".join(map(str, head_v))

    if head_v[0] > base_v[0]:
        print(f"✓ contract guard: {len(touched)} contract file(s) touched and "
              f"major version bumped {base_s} → {head_s}")
        for f in touched:
            print(f"    · {f}")
        return 0

    print(f"✖ Contract gate (issue #73): this PR touches the contract surface "
          f"without a MAJOR version bump ({base_s} → {head_s}).")
    print("   Touched contract files:")
    for f in touched:
        print(f"    · {f}")
    print("   Remediation / الإصلاح المطلوب:")
    print("   1. Bump the MAJOR version in pyproject.toml "
          f"(e.g. {base_s} → {base_v[0] + 1}.0.0) in this same PR.")
    print("   2. Update tests/golden_contract_signature.json in the same PR "
          "(re-capture the signature) with a one-line justification.")
    print("   3. For BREAKING changes (field removal/rename/type change, "
          "new required field): bump schema_version major in "
          "src/ocr_core/schemas/ocr_result.schema.json AND open an "
          "advance-notice issue linking the three consumer repos "
          "(arabic-handwriting-ocr-toolkit, arabic-medical-handwriting, "
          "medical-rag-ar) before merging.")
    print("   — أي تغيير لسطح العقد = رفع الإصدار الرئيسي + إشعار مسبق (بوابة #73)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
