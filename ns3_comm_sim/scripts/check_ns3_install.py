from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def read_required_cmake(ns3_root: Path) -> str | None:
    cmake_file = ns3_root / "CMakeLists.txt"
    if not cmake_file.exists():
        return None
    text = cmake_file.read_text(encoding="utf-8")
    match = re.search(r"cmake_minimum_required\(VERSION\s+([^\)]+)\)", text)
    if not match:
        return None
    return match.group(1).strip()


def current_cmake_version() -> str:
    result = subprocess.run(["cmake", "--version"], text=True, capture_output=True, check=True)
    first_line = result.stdout.splitlines()[0]
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", first_line)
    if not match:
        raise RuntimeError(f"Unable to parse CMake version from: {first_line}")
    return match.group(1)


def is_compatible(current: str, required: str | None) -> bool:
    if required is None:
        return False
    lower_bound = required.split("..")[0].strip()
    return parse_version(current) >= parse_version(lower_bound)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns3-install-root", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    install_root = Path(args.ns3_install_root)
    cmake_version = current_cmake_version()
    results = []
    for path in sorted(install_root.glob("ns-3.*")):
        if not path.is_dir():
            continue
        required = read_required_cmake(path)
        results.append(
            {
                "path": str(path),
                "required_cmake": required,
                "current_cmake": cmake_version,
                "usable": is_compatible(cmake_version, required),
                "has_wrapper": (path / "ns3").exists(),
            }
        )

    preferred = next((item for item in reversed(results) if item["usable"] and item["has_wrapper"]), None)
    payload = {
        "current_cmake": cmake_version,
        "install_root": str(install_root),
        "versions": results,
        "preferred": preferred,
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
