#!/usr/bin/env python3
"""Build a platform-specific archive containing the skill and every Python wheel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from install_runtime import ROOT, resolve_base_python, resolve_wheel, wheel_python, python_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", help="pip target platform, e.g. win_amd64 (default: this machine)")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "psytrainer-ml-offline.zip")
    parser.add_argument("--python", help="Select CPython 3.12-3.14; its version is recorded in bundle.json")
    parser.add_argument("--legacy", action="store_true", help="Include original INI engine dependencies (CPython 3.14)")
    args = parser.parse_args()
    wheel = resolve_wheel(None) if args.legacy else None
    base = resolve_base_python(args.python, wheel_python(wheel) if wheel else None)
    required = python_version(base)
    output = args.output.expanduser().absolute()
    if output.exists():
        parser.error(f"output already exists: {output}")
    with tempfile.TemporaryDirectory(prefix="psytrainer-bundle-") as temp:
        staging = Path(temp) / "psytrainer-ml"
        staging.mkdir()
        downloader = Path(temp) / "downloader"
        subprocess.run([*base, "-m", "venv", str(downloader)], check=True)
        py = downloader / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        target = []
        if args.platform:
            target = ["--platform", args.platform, "--python-version", f"{required[0]}.{required[1]}",
                      "--implementation", "cp", "--abi", f"cp{required[0]}{required[1]}"]
        subprocess.run([
            str(py), "-m", "pip", "download", "--only-binary=:all:", *target,
            "--dest", str(staging / "wheelhouse"), "-r", str(ROOT / "requirements.txt"), *([str(wheel)] if wheel else []),
        ], check=True)
        for name in ("README.md", "README.zh-CN.md", "SKILL.md", "NOTICE.md", "LICENSE", "requirements.txt"):
            shutil.copy2(ROOT / name, staging / name)
        for name in ("scripts", "config", "fixtures", "references", "agents", "tests", "vendor"):
            shutil.copytree(ROOT / name, staging / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "ml.ini"))
        (staging / "bundle.json").write_text(json.dumps({
            "schema": "psytrainer-bundle/v1", "profile": "legacy" if args.legacy else "pipeline",
            "python": list(required), "platform": args.platform or "native",
        }, indent=2) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after dependency resolution and every download succeed.
        archive = shutil.make_archive(str(Path(temp) / "complete"), "zip", temp, "psytrainer-ml")
        shutil.move(archive, output)
    print(f"Complete Python dependency bundle: {output}")
    print("The target machine must have matching Python and OS shared libraries installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
