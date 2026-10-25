#!/usr/bin/env python3
"""E26 evidence script — independent post-fix re-verification of PR #66.

RECONSTRUCTED 2026-10-25. The original E26 script was lost together with the
verification environment reset; this copy was rebuilt to re-run the exact
procedure whose numbers were published in the E26 review comment
(github.com/DrAbdulmalek/ocr-core#issuecomment-6368829627):

1. Adapter unit tests (tests/test_chandra_engine.py) on the fixed branch:
   expected 14 passed / 0 failed  (published: 14/14).
2. Full package suite (pytest, Py3.12, pytest 9.0.2):
   expected 518 passed / 7 skipped  (published: 518/7).
3. Reproduce the pre-fix defect on commit 8b601e8 via an isolated git
   worktree: the same adapter test file there must yield
   6 failed / 8 passed  (the 6 tests covering the closed hole fail on the
   pre-fix code — proving the tests actually guard the fix).
4. Integrity: origin/main must still be 0466788 (no merge), PR branch head
   must be da0c831 (review addition only).

Live chandra-ocr integration (vllm-without-server / hf-without-torch) is NOT
re-run here automatically; it is covered by the gated in-repo tests added by
this review:  CHANDRA_INTEGRATION=1 pytest tests/test_chandra_integration.py

Usage (repo checkout with the PR branch installed in a venv):

    python evidence/pr66/scripts/e26_chandra_postfix_recheck.py \
        [--repo PATH] [--json OUT.json] [--skip-worktree]

Exit code 0 only if every gate holds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RECONSTRUCTED_NOTE = (
    "Rebuilt 2026-10-25 after environment reset; re-runs the procedure whose "
    "numbers were published in issuecomment-6368829627."
)

EXPECTED = {
    "adapter_tests": {"passed": 14, "failed": 0},
    "full_suite": {"passed": 518, "skipped": 7},
    "prefix_repro": {"passed": 8, "failed": 6},
    "main_sha": "0466788028d5b9034dad640efda944e7289e86d9",
    "pr_head_sha": "da0c831bedeec91018064ea8e12f4967f74f6e01",
}

ADAPTER_TEST_FILE = "tests/test_chandra_engine.py"
PREFIX_COMMIT = "8b601e8"


def _run(cmd, cwd=None, timeout=1800):
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


def _pytest_counts(output: str) -> dict:
    """Parse a pytest summary line like '518 passed, 7 skipped in 120.3s'."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in counts:
        marker = f" {key}"
        if marker in output:
            tail = output.split(marker, 1)[1].split(",", 1)[0].split()
            if tail and tail[0].isdigit():
                counts[key] = int(tail[0])
    return counts


def run_adapter_tests(repo: Path) -> dict:
    proc = _run([sys.executable, "-m", "pytest", ADAPTER_TEST_FILE, "-q"], cwd=repo)
    counts = _pytest_counts(proc.stdout + proc.stderr)
    ok = (
        counts["passed"] == EXPECTED["adapter_tests"]["passed"]
        and counts["failed"] == EXPECTED["adapter_tests"]["failed"]
    )
    return {
        "step": "adapter_unit_tests",
        "ok": ok,
        "expected": EXPECTED["adapter_tests"],
        "observed": counts,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
    }


def run_full_suite(repo: Path) -> dict:
    proc = _run([sys.executable, "-m", "pytest", "-q"], cwd=repo, timeout=3600)
    counts = _pytest_counts(proc.stdout + proc.stderr)
    ok = (
        counts["passed"] == EXPECTED["full_suite"]["passed"]
        and counts["skipped"] == EXPECTED["full_suite"]["skipped"]
        and counts["failed"] == 0
        and counts["errors"] == 0
    )
    return {
        "step": "full_package_suite",
        "ok": ok,
        "expected": EXPECTED["full_suite"],
        "observed": counts,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
    }


def reproduce_prefix_defect(repo: Path) -> dict:
    """Run the adapter tests on pre-fix 8b601e8 in an isolated worktree."""
    with tempfile.TemporaryDirectory(prefix="e26-prefix-wt-") as tmp_name:
        wt = Path(tmp_name) / "wt"
        add = _run(
            ["git", "worktree", "add", "--detach", str(wt), PREFIX_COMMIT], cwd=repo
        )
        if add.returncode != 0:
            return {
                "step": "prefix_repro_worktree",
                "ok": False,
                "error": f"worktree add failed: {add.stderr.strip()[-300:]}",
            }
        try:
            # The worktree has no installed package; reuse the parent venv so
            # only the SOURCE TREE differs (that is the variable under test).
            env_cmd = [sys.executable, "-m", "pytest", ADAPTER_TEST_FILE, "-q"]
            proc = _run(env_cmd, cwd=wt)
            counts = _pytest_counts(proc.stdout + proc.stderr)
            ok = (
                counts["failed"] == EXPECTED["prefix_repro"]["failed"]
                and counts["passed"] == EXPECTED["prefix_repro"]["passed"]
            )
            return {
                "step": "prefix_repro_worktree",
                "ok": ok,
                "commit": PREFIX_COMMIT,
                "expected": EXPECTED["prefix_repro"],
                "observed": counts,
                "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
            }
        finally:
            _run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo)


def check_refs(repo: Path) -> dict:
    ls_remote = _run(["git", "ls-remote", "origin", "main"], cwd=repo)
    remote_main = ls_remote.stdout.split()[0] if ls_remote.stdout else None
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    ok = (
        remote_main == EXPECTED["main_sha"]
        and head == EXPECTED["pr_head_sha"]
    )
    return {
        "step": "ref_integrity",
        "ok": ok,
        "expected": {
            "origin_main": EXPECTED["main_sha"][:7],
            "pr_head": EXPECTED["pr_head_sha"][:7],
        },
        "observed": {"origin_main": (remote_main or "")[:7], "pr_head": head[:7]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="PR-branch checkout to verify")
    parser.add_argument("--json", dest="json_out", help="write the report as JSON")
    parser.add_argument(
        "--skip-worktree", action="store_true", help="skip the 8b601e8 worktree step"
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ADAPTER_TEST_FILE).is_file():
        print(f"FAIL: {ADAPTER_TEST_FILE} not found under {repo}")
        return 2
    if shutil.which("git") is None:
        print("FAIL: git not available")
        return 2

    report = {
        "artifact": "e26_chandra_postfix_recheck",
        "note": RECONSTRUCTED_NOTE,
        "repo": str(repo),
        "steps": [],
    }
    report["steps"].append(run_adapter_tests(repo))
    report["steps"].append(run_full_suite(repo))
    report["steps"].append(check_refs(repo))
    if not args.skip_worktree:
        report["steps"].append(reproduce_prefix_defect(repo))

    report["overall_ok"] = all(step.get("ok") for step in report["steps"])
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
