#!/usr/bin/env python3
"""
Merge multiple modular PDDL domain files into a single, validator-ready domain.

- De-duplicates :requirements, :types, :constants, :predicates, :functions
- Preserves both (:action ...) and (:durative-action ...) blocks, de-duplicated by name
- Keeps simple formatting; intended for validation/composition, not for lossless re-print

Usage:
  python merge_domains.py \
    --name tccn-core \
    --domains \
      ../../data/pddl/domain/common.pddl \
      ../../data/pddl/domain/navigation.pddl \
      ../../data/pddl/domain/inspection.pddl \
      ../../data/pddl/domain/computation.pddl \
      ../../data/pddl/domain/relay.pddl \
      ../../data/pddl/domain/logistics.pddl \
    --output ./tccn-core.pddl

If --output is not provided, the merged domain is printed to stdout.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

SECTION_PATTERNS = {
    "requirements": re.compile(r"\(:requirements\s+([^)]*)\)", re.IGNORECASE | re.MULTILINE),
}

# Simple extractors for multi-line balanced sections
SECTION_HEADERS = [":types", ":constants", ":predicates", ":functions"]
ACTION_HEADERS = [":action", ":durative-action"]


def _extract_balanced_blocks(text: str, header: str) -> List[str]:
    blocks: List[str] = []
    idx = 0
    while True:
        start = text.find(f"({header}", idx)
        if start == -1:
            break
        # scan for balanced parentheses from start
        depth = 0
        end = start
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = text[start:end]
        blocks.append(block)
        idx = end
    return blocks


def _extract_section_lines(text: str, header: str) -> List[str]:
    """Return inner lines of a section like (:types ...)."""
    blocks = _extract_balanced_blocks(text, header)
    lines: List[str] = []
    for b in blocks:
        inner = b[len(f"({header}") : -1].strip()  # strip '(:header' ... ')'
        # Keep original newlines; split and strip empty
        for line in inner.splitlines():
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _extract_requirements(text: str) -> List[str]:
    reqs: Set[str] = set()
    for m in SECTION_PATTERNS["requirements"].finditer(text):
        parts = m.group(1).split()
        reqs.update(parts)
    return sorted(reqs)


def _extract_actions(text: str) -> Dict[Tuple[str, str], str]:
    """Return mapping (kind, name) -> full block text for actions.
    kind in {":action", ":durative-action"}
    """
    actions: Dict[Tuple[str, str], str] = {}
    for kind in ACTION_HEADERS:
        for block in _extract_balanced_blocks(text, kind):
            # parse name: '(:action name' or '(:durative-action name'
            header_line = block.splitlines()[0]
            m = re.search(r"\(\s*" + re.escape(kind) + r"\s+([a-zA-Z0-9_-]+)", header_line)
            if not m:
                # final attempt: search within first line anyway
                m = re.search(r"\s+([a-zA-Z0-9_-]+)\b", header_line)
            if m:
                name = m.group(1)
                actions[(kind, name)] = block
    return actions


def merge_domains(name: str, paths: List[Path]) -> str:
    merged_requirements: Set[str] = set()
    merged_sections: Dict[str, List[str]] = {h: [] for h in SECTION_HEADERS}
    seen_line: Dict[str, Set[str]] = {h: set() for h in SECTION_HEADERS}

    merged_actions: Dict[Tuple[str, str], str] = {}

    for p in paths:
        txt = p.read_text()
        # requirements
        merged_requirements.update(_extract_requirements(txt))
        # sections
        for h in SECTION_HEADERS:
            for line in _extract_section_lines(txt, h):
                if line not in seen_line[h]:
                    seen_line[h].add(line)
                    merged_sections[h].append(line)
        # actions
        for key, block in _extract_actions(txt).items():
            merged_actions[key] = block  # last one wins (but names should be unique across modules)

    # minimal pretty print
    out: List[str] = []
    out.append(f"(define (domain {name})")
    if merged_requirements:
        out.append("    (:requirements " + " ".join(sorted(merged_requirements)) + ")")

    for h in SECTION_HEADERS:
        lines = merged_sections[h]
        if lines:
            out.append(f"\n    ({h}")
            for line in lines:
                out.append(f"        {line}")
            out.append("    )")

    # actions
    if merged_actions:
        out.append("")
        for (_, _), block in sorted(merged_actions.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            # indent block by 4 spaces
            indented = "\n".join(["    "+ln if i>0 else "    "+ln for i, ln in enumerate(block.splitlines())])
            out.append(indented)
        out.append("")

    out.append(")")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Merge modular PDDL domains into a single domain")
    parser.add_argument("--name", default="tccn-merged", help="Name of the merged domain")
    parser.add_argument("--domains", nargs="+", type=Path, required=True, help="List of domain .pddl files in merge order")
    parser.add_argument("--output", type=Path, help="Output path for merged domain. If omitted, prints to stdout")
    args = parser.parse_args()

    merged = merge_domains(args.name, args.domains)
    if args.output:
        args.output.write_text(merged)
        print(f"Merged domain written to: {args.output}")
    else:
        print(merged)


if __name__ == "__main__":
    main()
