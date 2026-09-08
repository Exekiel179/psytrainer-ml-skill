#!/usr/bin/env python3
"""Install psytrainer-ml runtime dependencies at Skill install time.

Creates a local .venv under the skill root, installs requirements.txt, and
optionally installs a PsyTrainer wheel. Records the interpreter in runtime.json
so training/prediction use the install-time environment instead of ad-hoc pip.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
RUNTIME_JSON = ROOT / "runtime.json"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_base_python(preferred: str | None) -> str:
    if preferred:
        path = shutil.which(preferred) or preferred
        if not Path(path).exists() and shutil.which(preferred) is None:
            raise SystemExit(f"python not found: {preferred}")
        return path if Path(path).exists() else shutil.which(preferred)  # type: ignore[return-value]
    for candidate in ("python3.13", "python3.12", "python3.11", "python3"):
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("no suitable python3 interpreter found on PATH")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv(base_python: str) -> Path:
    if not VENV_DIR.exists():
        print(f"creating venv with {base_python} -> {VENV_DIR}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
    py = venv_python()
    if not py.is_file():
        raise SystemExit(f"venv python missing: {py}")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    return py


def install_requirements(py: Path) -> None:
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"missing {REQUIREMENTS}")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def install_wheel(py: Path, wheel: Path) -> None:
    if not wheel.is_file():
        raise SystemExit(f"PsyTrainer wheel not found: {wheel}")
    run([str(py), "-m", "pip", "install", "--force-reinstall", str(wheel)])


def verify(py: Path, require_psytrainer: bool) -> dict[str, object]:
    code = (
        "import importlib.util, json, sys\n"
        "mods = ['pandas', 'numpy', 'joblib', 'ccpl_training_models']\n"
        "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))\n"
    )
    out = subprocess.check_output([str(py), "-c", code], text=True).strip()
    status = json.loads(out)
    print("import probe:", status, flush=True)
    for name in ("pandas", "numpy", "joblib"):
        if not status.get(name):
            raise SystemExit(f"required module missing after install: {name}")
    if require_psytrainer and not status.get("ccpl_training_models"):
        raise SystemExit(
            "ccpl_training_models missing; pass --wheel /path/to/PsyTrainer-*.whl "
            "or set PSYTRAINER_WHEEL"
        )
    return status


def write_runtime(py: Path, status: dict[str, object], wheel: str | None) -> None:
    payload = {
        "schemaVersion": "psytrainer-ml/runtime/v1",
        "python": str(py.resolve()),
        "venv": str(VENV_DIR.resolve()),
        "imports": status,
        "wheel": wheel,
    }
    RUNTIME_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RUNTIME_JSON}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install psytrainer-ml runtime at Skill install time")
    parser.add_argument("--python", help="Base Python used to create .venv (prefer 3.13 for PsyTrainer wheels)")
    parser.add_argument("--wheel", help="Path to PsyTrainer-*.whl (or set PSYTRAINER_WHEEL)")
    parser.add_argument(
        "--allow-missing-psytrainer",
        action="store_true",
        help="Install pandas/numpy/joblib even if the PsyTrainer wheel is unavailable",
    )
    args = parser.parse_args(argv)

    wheel_env = os.environ.get("PSYTRAINER_WHEEL", "").strip()
    wheel_path = Path(args.wheel).expanduser() if args.wheel else (Path(wheel_env).expanduser() if wheel_env else None)

    base = resolve_base_python(args.python)
    py = ensure_venv(base)
    install_requirements(py)
    if wheel_path is not None:
        install_wheel(py, wheel_path)
    elif not args.allow_missing_psytrainer:
        print(
            "No --wheel / PSYTRAINER_WHEEL provided. "
            "Re-run with the vendor PsyTrainer wheel to finish install.",
            file=sys.stderr,
            flush=True,
        )
        status = verify(py, require_psytrainer=False)
        write_runtime(py, status, None)
        return 2

    status = verify(py, require_psytrainer=not args.allow_missing_psytrainer)
    write_runtime(py, status, str(wheel_path.resolve()) if wheel_path else None)
    print("psytrainer-ml runtime install complete", flush=True)
    return 0 if status.get("ccpl_training_models") or args.allow_missing_psytrainer else 2


if __name__ == "__main__":
    raise SystemExit(main())
