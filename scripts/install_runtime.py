#!/usr/bin/env python3
"""Install psytrainer-ml runtime dependencies at Skill install time.

Cross-platform (macOS / Linux / Windows):
  - Creates a local .venv under the skill root
  - Installs requirements.txt
  - Optionally installs a PsyTrainer wheel (--wheel / PSYTRAINER_WHEEL; globs OK)
  - Writes runtime.json so task runs use the install-time interpreter (no mid-task pip)

Windows notes:
  - Prefers the `py` launcher (py -3.13 …) then `python`
  - venv interpreter is .venv\\Scripts\\python.exe
  - Wheel globs are expanded in Python (cmd.exe does not expand *)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
RUNTIME_JSON = ROOT / "runtime.json"
IS_WINDOWS = os.name == "nt"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = subprocess.list2cmdline(cmd) if IS_WINDOWS else " ".join(cmd)
    print("+", printable, flush=True)
    subprocess.run(cmd, check=True, env=env)


def python_version(cmd: list[str]) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [*cmd, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)", out)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_base_python(preferred: str | None) -> list[str]:
    """Return an argv prefix that launches the base interpreter (may be `py -3.13`)."""
    if preferred:
        # Allow "py -3.13" as a single quoted string, or a bare path/name.
        parts = preferred.split() if preferred.lower().startswith("py ") else [preferred]
        if len(parts) == 1:
            resolved = shutil.which(parts[0]) or parts[0]
            parts = [resolved]
        if python_version(parts) is None:
            raise SystemExit(f"python not runnable: {preferred}")
        return parts

    candidates: list[list[str]] = []
    if IS_WINDOWS:
        for minor in ("3.13", "3.12", "3.11", "3.10"):
            candidates.append(["py", f"-{minor}"])
        candidates.append(["py", "-3"])
        for name in ("python3.13", "python3.12", "python3.11", "python", "python3"):
            found = shutil.which(name)
            if found:
                candidates.append([found])
    else:
        for name in ("python3.13", "python3.12", "python3.11", "python3"):
            found = shutil.which(name)
            if found:
                candidates.append([found])

    scored: list[tuple[tuple[int, int], list[str]]] = []
    for cmd in candidates:
        version = python_version(cmd)
        if version and version[0] >= 3 and version[1] >= 10:
            scored.append((version, cmd))
    if not scored:
        raise SystemExit(
            "no suitable Python 3.10+ interpreter found on PATH"
            + (" (try: py -3.13 or install Python from python.org)" if IS_WINDOWS else "")
        )
    # Prefer higher version (PsyTrainer wheels are often cp313).
    scored.sort(key=lambda item: item[0], reverse=True)
    best_version, best_cmd = scored[0]
    print(f"base python: {' '.join(best_cmd)} ({best_version[0]}.{best_version[1]})", flush=True)
    return best_cmd


def venv_python() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def venv_pip() -> Path:
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def ensure_venv(base_cmd: list[str], *, recreate: bool) -> Path:
    if recreate and VENV_DIR.exists():
        print(f"removing existing venv {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        printable = " ".join(base_cmd)
        print(f"creating venv with {printable} -> {VENV_DIR}", flush=True)
        # Use the selected interpreter to create the venv so Windows `py -3.13` works.
        run([*base_cmd, "-m", "venv", str(VENV_DIR)])
    py = venv_python()
    if not py.is_file():
        raise SystemExit(f"venv python missing: {py}")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    return py


def resolve_wheel(spec: str | None) -> Path | None:
    """Resolve a wheel path; expand globs in-process (needed on Windows cmd)."""
    if not spec:
        return None
    raw = spec.strip().strip('"').strip("'")
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()

    def collect(pattern: str) -> list[Path]:
        found: list[Path] = []
        candidate = Path(pattern).expanduser()
        if candidate.is_file():
            return [candidate.resolve()]
        # Absolute or relative globs: glob only the final name under the parent.
        parent = candidate.parent if str(candidate.parent) not in ("", ".") else Path.cwd()
        name = candidate.name
        try:
            if parent.exists():
                found.extend(parent.glob(name))
        except OSError:
            pass
        try:
            found.extend(ROOT.glob(name))
        except OSError:
            pass
        if not candidate.is_absolute():
            try:
                found.extend(Path.cwd().glob(pattern))
            except (OSError, NotImplementedError, ValueError):
                pass
        return found

    matches: list[Path] = []
    for pattern in (raw, str(ROOT / raw)):
        matches.extend(collect(pattern))

    uniq: list[Path] = []
    seen: set[Path] = set()
    for item in matches:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in seen:
            seen.add(resolved)
            uniq.append(resolved)
    if not uniq:
        raise SystemExit(f"PsyTrainer wheel not found for pattern: {spec}")
    if len(uniq) > 1:
        listing = "\n".join(f"  - {item}" for item in uniq)
        raise SystemExit(f"multiple wheels matched {spec!r}; pass one path:\n{listing}")
    return uniq[0]


def install_requirements(py: Path) -> None:
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"missing {REQUIREMENTS}")
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def install_wheel(py: Path, wheel: Path) -> None:
    run([str(py), "-m", "pip", "install", "--force-reinstall", str(wheel)])


def verify(py: Path, require_psytrainer: bool) -> dict[str, object]:
    code = (
        "import importlib.util, json\n"
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
            "ccpl_training_models missing; pass --wheel path\\to\\PsyTrainer-*.whl "
            "or set PSYTRAINER_WHEEL"
        )
    return status


def write_runtime(py: Path, status: dict[str, object], wheel: str | None, base_cmd: list[str]) -> None:
    payload = {
        "schemaVersion": "psytrainer-ml/runtime/v1",
        "platform": os.name,
        "python": str(py.resolve()),
        "pythonLauncher": base_cmd,
        "venv": str(VENV_DIR.resolve()),
        "pip": str(venv_pip().resolve()) if venv_pip().exists() else None,
        "imports": status,
        "wheel": wheel,
        "windows": IS_WINDOWS,
    }
    RUNTIME_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RUNTIME_JSON}", flush=True)
    if IS_WINDOWS:
        print(f"Windows interpreter: {py}", flush=True)
        print("Task runs should use runtime.json python (Scripts\\python.exe).", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install psytrainer-ml runtime at Skill install time (Windows/macOS/Linux)"
    )
    parser.add_argument(
        "--python",
        help='Base Python for the venv, e.g. "py -3.13", C:\\Python313\\python.exe, or python3.13',
    )
    parser.add_argument(
        "--wheel",
        help="Path or glob to PsyTrainer-*.whl (or set PSYTRAINER_WHEEL). Globs work on Windows.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate .venv before installing",
    )
    parser.add_argument(
        "--allow-missing-psytrainer",
        action="store_true",
        help="Install pandas/numpy/joblib even if the PsyTrainer wheel is unavailable",
    )
    args = parser.parse_args(argv)

    wheel_spec = (args.wheel or os.environ.get("PSYTRAINER_WHEEL", "")).strip() or None
    wheel_path = resolve_wheel(wheel_spec)

    base_cmd = resolve_base_python(args.python)
    py = ensure_venv(base_cmd, recreate=args.recreate)
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
        write_runtime(py, status, None, base_cmd)
        return 2

    status = verify(py, require_psytrainer=not args.allow_missing_psytrainer)
    write_runtime(py, status, str(wheel_path.resolve()) if wheel_path else None, base_cmd)
    print("psytrainer-ml runtime install complete", flush=True)
    return 0 if status.get("ccpl_training_models") or args.allow_missing_psytrainer else 2


if __name__ == "__main__":
    raise SystemExit(main())
