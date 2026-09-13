#!/usr/bin/env python3
"""Install psytrainer-ml runtime dependencies at Skill install time.

Cross-platform (macOS / Linux / Windows):
  - Creates a local .venv under the skill root
  - Installs requirements.txt
  - Installs the bundled PsyTrainer wheel and all transitive dependencies
  - Writes runtime.json so task runs use the install-time interpreter (no mid-task pip)

Windows notes:
  - Selects the Python version declared by the wheel
  - venv interpreter is .venv\\Scripts\\python.exe
  - Wheel globs are expanded in Python (cmd.exe does not expand *)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
RUNTIME_JSON = ROOT / "runtime.json"
IS_WINDOWS = os.name == "nt"
BUNDLED_WHEEL = ROOT / "vendor" / "PsyTrainer-0.2.0-cp314-none-any.whl"
BUNDLED_SHA256 = "3e999045342b82cf25cb453e61a185530b553e4b9d439d2c5f5964fc5e55c357"


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


def resolve_base_python(preferred: str | None, required: tuple[int, int]) -> list[str]:
    """Return an argv prefix that launches a matching base interpreter."""
    if preferred:
        # Allow "py -3.14" as a single quoted string, or a bare path/name.
        parts = preferred.split() if preferred.lower().startswith("py ") else [preferred]
        if len(parts) == 1:
            resolved = shutil.which(parts[0]) or parts[0]
            parts = [resolved]
        if python_version(parts) != required:
            raise SystemExit(f"Python {required[0]}.{required[1]} required by wheel: {preferred}")
        return parts

    version_name = f"{required[0]}.{required[1]}"
    candidates: list[list[str]] = [[getattr(sys, "_base_executable", sys.executable)]]
    if IS_WINDOWS:
        candidates.append(["py", f"-{version_name}"])
    for name in (f"python{version_name}", "python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    if shutil.which("uv"):
        try:
            found = subprocess.check_output(
                ["uv", "python", "find", "--no-python-downloads", version_name],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            candidates.append([found])
        except subprocess.CalledProcessError:
            pass

    scored: list[tuple[tuple[int, int], list[str]]] = []
    for cmd in candidates:
        version = python_version(cmd)
        if version == required:
            scored.append((version, cmd))
    if not scored:
        raise SystemExit(
            f"Python {version_name} required by the PsyTrainer wheel. "
            f"Install it from python.org or run: uv python install {version_name}"
        )
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
    if VENV_DIR.exists() and not recreate:
        if python_version([str(venv_python())]) != python_version(base_cmd):
            raise SystemExit("existing .venv uses a different Python; re-run with --recreate")
    if recreate and VENV_DIR.exists():
        print(f"removing existing venv {VENV_DIR}", flush=True)
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        printable = " ".join(base_cmd)
        print(f"creating venv with {printable} -> {VENV_DIR}", flush=True)
        # Use the selected interpreter so Windows launcher arguments work too.
        run([*base_cmd, "-m", "venv", str(VENV_DIR)])
    py = venv_python()
    if not py.is_file():
        raise SystemExit(f"venv python missing: {py}")
    return py


def resolve_wheel(spec: str | None) -> Path:
    """Resolve a wheel path; expand globs in-process (needed on Windows cmd)."""
    if not spec:
        if not BUNDLED_WHEEL.is_file():
            raise SystemExit(f"incomplete download: bundled wheel missing: {BUNDLED_WHEEL}")
        if hashlib.sha256(BUNDLED_WHEEL.read_bytes()).hexdigest() != BUNDLED_SHA256:
            raise SystemExit("bundled PsyTrainer wheel checksum mismatch; download the complete package again")
        return BUNDLED_WHEEL
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


def wheel_python(wheel: Path) -> tuple[int, int]:
    """Read the wheel's declared tag without guessing from a filename."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            if archive.testzip() is not None:
                raise ValueError("corrupt wheel archive")
            metadata_files = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            wheel_files = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
            if len(metadata_files) != 1 or len(wheel_files) != 1:
                raise ValueError("missing or ambiguous wheel metadata")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
            if metadata.get("Name", "").lower() != "psytrainer":
                raise ValueError("expected PsyTrainer distribution")
            tags = BytesParser().parsebytes(archive.read(wheel_files[0])).get_all("Tag", [])
            versions = {match.groups() for tag in tags
                        if (match := re.fullmatch(r"cp(3)(\d+)-none-any", tag))}
            if len(versions) != 1:
                raise ValueError(f"expected a single CPython purelib tag, got {tags}")
            major, minor = versions.pop()
            return int(major), int(minor)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise SystemExit(f"invalid PsyTrainer wheel: {exc}") from exc


def install_requirements(py: Path, wheel: Path, wheelhouse: Path | None = None) -> None:
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"missing {REQUIREMENTS}")
    options = ["--no-index", "--find-links", str(wheelhouse)] if wheelhouse else []
    # Resolve wrapper and vendor requirements together so neither overwrites the other.
    run([str(py), "-m", "pip", "install", *options, "-r", str(REQUIREMENTS), str(wheel)])
    run([str(py), "-m", "pip", "check"])


def verify(py: Path) -> dict[str, object]:
    code = (
        "import importlib, json\n"
        "mods = ['pandas', 'numpy', 'joblib', 'ccpl_training_models']\n"
        "status = {}\n"
        "for m in mods:\n"
        "    importlib.import_module(m)\n"
        "    status[m] = True\n"
        "from ccpl_training_models.trainer import Trainer\n"
        "trainer = Trainer('general')\n"
        "assert all(callable(getattr(trainer, m, None)) for m in ['set_base_config', 'set_scoring', 'run'])\n"
        "print(json.dumps(status))\n"
    )
    out = subprocess.check_output([str(py), "-c", code], text=True).strip()
    status = json.loads(out.splitlines()[-1])
    print("import probe:", status, flush=True)
    for name in ("pandas", "numpy", "joblib", "ccpl_training_models"):
        if not status.get(name):
            raise SystemExit(f"required module missing after install: {name}")
    return status


def write_runtime(py: Path, status: dict[str, object], wheel: str | None, base_cmd: list[str]) -> None:
    payload = {
        "schemaVersion": "psytrainer-ml/runtime/v1",
        "ready": True,
        "platform": os.name,
        "python": str(py.absolute()),
        "pythonLauncher": base_cmd,
        "venv": str(VENV_DIR.resolve()),
        "pip": str(venv_pip().resolve()) if venv_pip().exists() else None,
        "imports": status,
        "wheel": wheel,
        "windows": IS_WINDOWS,
    }
    temporary = RUNTIME_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RUNTIME_JSON)
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
        help='Base Python matching the wheel, e.g. "py -3.14" or python3.14',
    )
    parser.add_argument(
        "--wheel",
        help="Override the bundled PsyTrainer wheel (or set PSYTRAINER_WHEEL). Globs work on Windows.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate .venv before installing",
    )
    parser.add_argument(
        "--allow-missing-psytrainer",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--wheelhouse", type=Path, help="Install all dependencies offline from this directory")
    args = parser.parse_args(argv)

    # A failed re-install must never leave an old success marker for callers.
    RUNTIME_JSON.unlink(missing_ok=True)
    if args.allow_missing_psytrainer:
        raise SystemExit("--allow-missing-psytrainer is no longer supported: a complete runtime is required")
    wheel_spec = (args.wheel or os.environ.get("PSYTRAINER_WHEEL", "")).strip() or None
    wheel_path = resolve_wheel(wheel_spec)
    required = wheel_python(wheel_path)
    wheelhouse = args.wheelhouse
    if wheelhouse is None and (ROOT / "wheelhouse").is_dir():
        wheelhouse = ROOT / "wheelhouse"
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse directory not found: {wheelhouse}")
    base_cmd = resolve_base_python(args.python, required)
    py = ensure_venv(base_cmd, recreate=args.recreate)
    install_requirements(py, wheel_path, wheelhouse)
    status = verify(py)
    write_runtime(py, status, str(wheel_path.resolve()), base_cmd)
    print("psytrainer-ml runtime install complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
