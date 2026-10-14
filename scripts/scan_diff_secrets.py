#!/usr/bin/env python3
"""Secret Gate (S1-T5): scan ADDED lines of a unified diff for credential-like material.

Scope: credential detection only (PATs, API keys, private keys, credentialed URLs,
bot tokens). Sensitive-content policy (medical data, Telegram sessions, pirated
content) is a separate, non-regex class and is NOT covered here.

Output is REDACTED BY DESIGN: reports file, new-file line number and pattern
class. Never prints the matched value. Exit 1 on any finding, 0 otherwise.
"""
import re
import sys

# Never scan this script's own diff hunks (it necessarily contains pattern text).
SELF_NAME = "scan_diff_secrets.py"

# Allowlist: example/template fixtures whose context line is an obvious placeholder.
PLACEHOLDER_CTX = re.compile(
    r"(?i)(xxx|example|your[-_]|changeme|placeholder|dummy|sample|"
    r"token[_-]?here|<[^>]+>|\$\{|REDACTED|fake)"
)
ALLOWLIST_PATHS = (
    ".env.example",
    ".env.web.example",
    "settings.env.example",
)

PATTERNS = {
    "github-classic-pat": re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "github-fine-pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "github-oauth": re.compile(r"gho_[A-Za-z0-9]{36}"),
    "openai-style-key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "aws-access-key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "slack-token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    "telegram-bot-token": re.compile(r"\b\d{8,10}:AA[A-Za-z0-9_-]{33}\b"),
    "private-key-block": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"
    ),
    "cred-with-scheme-url": re.compile(
        r"[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@"
    ),
}


def scan(diff_text):
    findings = []
    new_file = None
    new_line = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            new_file = raw[6:]
            continue
        if raw.startswith("+++"):
            new_file = None
            continue
        if raw.startswith("---") or raw.startswith("diff ") or raw.startswith("index "):
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", raw)
        if m:
            new_line = int(m.group(1))
            continue
        if new_file is None:
            continue
        if raw.startswith("+"):
            body = raw[1:]
            hit = False
            if SELF_NAME not in new_file:
                for name, pat in PATTERNS.items():
                    mm = pat.search(body)
                    if mm:
                        # placeholder guard: benign example lines stay clean
                        if not PLACEHOLDER_CTX.search(body):
                            findings.append((new_file, new_line, name))
                            hit = True
            new_line += 1
        elif raw.startswith(" ") or raw.startswith("-"):
            if not raw.startswith("-"):
                new_line += 1
    return findings


def main():
    if len(sys.argv) > 1:
        diff_text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    else:
        diff_text = sys.stdin.read()
    findings = scan(diff_text)
    if findings:
        print(f"SECRET GATE: BLOCKED — {len(findings)} credential-like finding(s)")
        for path, line, cls in findings:
            print(f"  {path}:{line}: {cls}  (value redacted by design)")
        print("Remove the secret, move it to a secret store, or use a placeholder.")
        return 1
    print("SECRET GATE: CLEAN — no credential-like material in added lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
