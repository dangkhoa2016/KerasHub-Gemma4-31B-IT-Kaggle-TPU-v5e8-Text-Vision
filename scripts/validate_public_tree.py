#!/usr/bin/env python3
"""Source-level validation of the public repository tree.

This is a static guard. It needs no TPU, no network access and no model
weights, so it can run in lightweight CI and in every historical commit.

It fails when:

1. a public artifact references a repository path that does not exist;
2. an internal development token remains in a public artifact;
3. a required English/Vietnamese documentation pair is missing;
4. a local Markdown link does not resolve.

Both the token deny-list and the required document pairs are explicit lists
rather than broad regular expressions, so the check does not produce false
positives on ordinary prose.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".cfg",
    ".ipynb",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
SKIP_DIRECTORIES = {".git", ".ipynb_checkpoints", "logs", "state"}

# This file necessarily spells out the denied tokens in order to match them, so
# it is exempt from the token scan only. Path, link and bilingual checks still
# cover it.
TOKEN_SCAN_EXEMPT = {"scripts/validate_public_tree.py"}

# Directories whose Markdown documents must ship an English/Vietnamese pair.
BILINGUAL_DIRECTORIES = ("clients", "docs", "evidence")

# Public documents that live at the repository root.
BILINGUAL_ROOT_DOCUMENTS = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
)

# Repository top-level directories a public artifact may point at.
PUBLIC_PATH_ROOTS = (
    "clients",
    "docs",
    "evidence",
    "notebooks",
    "scripts",
    "src",
    "tests",
)

PUBLIC_PATH_RE = re.compile(
    r"(?<![\w./-])((?:" + "|".join(PUBLIC_PATH_ROOTS) + r")/[\w./-]*)"
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
URL_RE = re.compile(r"https?://\S+")

FORBIDDEN_TOKENS = (
    (re.compile(r"\bG(?:7|8|9|10)\b"), "internal stage label"),
    (re.compile(r"\bG(?:7|8|9|10)[A-Z_]"), "internal stage label"),
    (re.compile(r"\bg(?:7|8|9|10)[-_]"), "internal stage label"),
    (re.compile(r"\b_g(?:7|8|9|10)_"), "internal stage marker"),
    (re.compile(r"CLOSED/PASS"), "internal adjudication label"),
    (re.compile(r"final_tpu_one_shot"), "removed internal orchestrator"),
    (re.compile(r"\bsuperpowers\b"), "internal tooling name"),
    (re.compile(r"internal gate", re.IGNORECASE), "internal process label"),
    (re.compile(r"authority handoff", re.IGNORECASE), "internal process label"),
)

# Known internal markers that are deliberately still present. Each entry is
# (path, token description, reason). These are reported as warnings so the
# deferred cleanup stays visible instead of silently accepted.
ALLOWED_TOKENS = (
    (
        "src/gemma4_server/tpu/observability.py",
        "internal stage marker",
        "handler sentinel in the frozen qualified runtime; renaming it would "
        "change the published src tree, so the cleanup is deferred",
    ),
    (
        "tests/test_observability.py",
        "internal stage marker",
        "asserts that the src handler sentinel is detached after use",
    ),
)


def public_files() -> list[Path]:
    found = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            found.append(path)
    return found


def notebook_text(path: Path) -> list[tuple[str, str]]:
    """Return (cell kind, text) pairs so a notebook is checked like a document."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, list):
            body = "".join(source)
        else:
            body = str(source)
        cells.append((f"cell {index} ({cell.get('cell_type')})", body))
    return cells


def document_chunks(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".ipynb":
        return notebook_text(path)
    return [("document", path.read_text(encoding="utf-8", errors="replace"))]


def check_referenced_paths(errors: list[str]) -> None:
    for path in public_files():
        for label, text in document_chunks(path):
            stripped = URL_RE.sub(" ", text)
            for match in PUBLIC_PATH_RE.finditer(stripped):
                reference = match.group(1).rstrip("./")
                if not (ROOT / reference).exists():
                    errors.append(
                        f"{path.relative_to(ROOT)}: {label} references a missing "
                        f"repository path: {reference}"
                    )


def check_markdown_links(errors: list[str]) -> None:
    for path in public_files():
        if path.suffix not in {".md", ".ipynb"}:
            continue
        for label, text in document_chunks(path):
            for target in MARKDOWN_LINK_RE.findall(URL_RE.sub(" ", text)):
                if target.startswith(("#", "mailto:")):
                    continue
                local = target.split("#", 1)[0]
                if not local:
                    continue
                if not (path.parent / local).resolve().exists():
                    errors.append(
                        f"{path.relative_to(ROOT)}: {label} has a broken local "
                        f"link: {target}"
                    )


def check_bilingual_pairs(errors: list[str]) -> None:
    for name in BILINGUAL_ROOT_DOCUMENTS:
        if not (ROOT / f"{name[:-3]}.vi.md").is_file():
            errors.append(f"missing Vietnamese counterpart for {name}")
    for directory in BILINGUAL_DIRECTORIES:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            if path.name.endswith(".vi.md"):
                continue
            counterpart = path.with_name(f"{path.stem}.vi.md")
            if not counterpart.is_file():
                errors.append(
                    f"missing Vietnamese counterpart: "
                    f"{path.relative_to(ROOT)} -> {counterpart.relative_to(ROOT)}"
                )


def check_forbidden_tokens(errors: list[str], warnings: list[str]) -> None:
    allowed = {(name, kind) for name, kind, _ in ALLOWED_TOKENS}
    reasons = {name: reason for name, _, reason in ALLOWED_TOKENS}
    for path in public_files():
        relative = str(path.relative_to(ROOT))
        if relative in TOKEN_SCAN_EXEMPT:
            continue
        for label, text in document_chunks(path):
            for pattern, kind in FORBIDDEN_TOKENS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    where = f"{relative}: {label} line {line}: {kind} {match.group(0)!r}"
                    if (relative, kind) in allowed:
                        warnings.append(f"{where} (allowed: {reasons[relative]})")
                    else:
                        errors.append(where)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    check_referenced_paths(errors)
    check_markdown_links(errors)
    check_bilingual_pairs(errors)
    check_forbidden_tokens(errors, warnings)

    for warning in warnings:
        print(f"WARNING {warning}")
    for error in errors:
        print(f"ERROR {error}")

    checks = (
        "notebook and artifact path references",
        "local markdown links",
        "bilingual documentation pairs",
        "internal token deny-list",
    )
    if errors:
        print(f"public tree validation FAILED with {len(errors)} problem(s)")
        for name in checks:
            print(f"  - {name}")
        return 1
    print("public tree validation PASSED")
    for name in checks:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
