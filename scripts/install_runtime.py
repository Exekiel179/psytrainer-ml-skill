#!/usr/bin/env python3
"""Install psytrainer-ml runtime dependencies at Skill install time.

Cross-platform (macOS / Linux / Windows):
  - Creates a local .venv under the skill root
  - Installs requirements.txt
  - Installs the complete local Pipeline runtime on CPython 3.12-3.14
  - Optionally installs PsyTrainer in a separate .venv-legacy
  - Writes runtime.json so task runs use the install-time interpreter (no mid-task pip)

Windows notes:
  - Legacy mode selects the Python version declared by the wheel
  - venv interpreter is .venv\\Scripts\\python.exe
  - Wheel globs are expanded in Python (cmd.exe does not expand *)
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
SUPPORTED_PYTHONS = ((3, 12), (3, 13), (3, 14))


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = subprocess.list2cmdline(cmd) if IS_WINDOWS else " ".join(cmd)
    print("+", printable, flush=True)
    subprocess.run(cmd, check=True, env=env)


def python_version(cmd: list[str]) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [*cmd, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}' if sys.implementation.name == 'cpython' else 'unsupported')"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.fullmatch(r"(\d+)\.(\d+)", out)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_base_python(preferred: str | None, required: tuple[int, int] | None = None) -> list[str]:
    """Return an argv prefix that launches a matching base interpreter."""
    allowed = (required,) if required else SUPPORTED_PYTHONS
    requirement = f"{required[0]}.{required[1]} required by wheel" if required else "3.12-3.14 required for Pipeline"
    if preferred:
        # Allow "py -3.14" as a single quoted string, or a bare path/name.
        parts = preferred.split() if preferred.lower().startswith("py ") else [preferred]
        if len(parts) == 1:
            resolved = shutil.which(parts[0]) or parts[0]
            parts = [resolved]
        if python_version(parts) not in allowed:
            raise SystemExit(f"CPython {requirement}: {preferred}")
        return parts

    version_names = [f"{major}.{minor}" for major, minor in allowed]
    candidates: list[list[str]] = []
    if venv_python().is_file():
        candidates.append([str(venv_python())])
    candidates.append([getattr(sys, "_base_executable", sys.executable)])
    if IS_WINDOWS:
        candidates.extend([["py", f"-{version}"] for version in version_names])
    for name in (*[f"python{version}" for version in version_names], "python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append([found])
    if shutil.which("uv"):
        for version_name in version_names:
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
        if version in allowed:
            scored.append((version, cmd))
    if not scored:
        raise SystemExit(
            f"CPython {requirement}. "
            f"Install it from python.org or run: uv python install {version_names[0]}"
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
    # Resolve a venv launcher to its base before deleting that venv on recreation.
    if recreate and VENV_DIR.exists():
        base = subprocess.check_output([*base_cmd, "-c", "import sys; print(sys._base_executable)"], text=True).strip()
        base_cmd = [base]
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
        raise SystemExit("PsyTrainer is not distributed with this Skill. For existing INI jobs only, "
                         "provide --legacy --wheel /path/to/PsyTrainer.whl; new analyses use Pipeline.")
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


def install_requirements(py: Path, wheel: Path | None = None, wheelhouse: Path | None = None) -> None:
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"missing {REQUIREMENTS}")
    options = ["--no-index", "--find-links", str(wheelhouse)] if wheelhouse else []
    # Resolve wrapper and vendor requirements together so neither overwrites the other.
    run([str(py), "-m", "pip", "install", *options, "-r", str(REQUIREMENTS), *([str(wheel)] if wheel else [])])
    run([str(py), "-m", "pip", "check"])


def verify(py: Path, *, legacy: bool = False) -> dict[str, object]:
    modules = ["pandas", "numpy", "scipy", "joblib", "sklearn", "imblearn", "matplotlib", "docx",
               "lightgbm", "xgboost", "catboost"]
    if legacy:
        modules.append("ccpl_training_models")
    code = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        f"mods = {modules!r}\n"
        "status = {}\n"
        "for m in mods:\n"
        "    importlib.import_module(m)\n"
        "    status[m] = True\n"
        "from model_registry import MODELS, create_estimator\n"
        "for task, models in MODELS.items():\n"
        "    for tag in models:\n"
        "        model = create_estimator(tag, task, 42)\n"
        "        assert callable(model.fit) and callable(model.predict)\n"
    )
    if legacy:
        code += (
            "from ccpl_training_models.trainer import Trainer\n"
            "trainer = Trainer('general')\n"
            "assert all(callable(getattr(trainer, m, None)) for m in ['set_base_config', 'set_scoring', 'run'])\n"
        )
    code += "print(json.dumps(status))\n"
    out = subprocess.check_output([str(py), "-c", code], text=True).strip()
    status = json.loads(out.splitlines()[-1])
    print("import probe:", status, flush=True)
    for name in modules:
        if not status.get(name):
            raise SystemExit(f"required module missing after install: {name}")
    return status


def write_runtime(py: Path, status: dict[str, object], wheel: str | None, base_cmd: list[str]) -> None:
    payload = {
        "schemaVersion": "psytrainer-ml/runtime/v2",
        "profile": "legacy" if wheel else "pipeline",
        "capabilities": {"pipeline": True, "legacy": bool(wheel)},
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
        print(f"Task runs should use {RUNTIME_JSON.name} python (Scripts\\python.exe).", flush=True)


@contextmanager
def runtime_location(legacy: bool):
    global VENV_DIR, RUNTIME_JSON
    previous = VENV_DIR, RUNTIME_JSON
    if legacy:
        VENV_DIR, RUNTIME_JSON = ROOT / ".venv-legacy", ROOT / "runtime-legacy.json"
    try:
        yield
    finally:
        VENV_DIR, RUNTIME_JSON = previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install psytrainer-ml runtime at Skill install time (Windows/macOS/Linux)"
    )
    parser.add_argument(
        "--python",
        help='CPython 3.12-3.14 for Pipeline; wheel-matching Python for --legacy',
    )
    parser.add_argument(
        "--wheel",
        help="Explicit external PsyTrainer wheel for existing INI jobs only. Globs work on Windows.",
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
    parser.add_argument("--legacy", action="store_true", help="Existing INI compatibility only; requires an external --wheel")
    args = parser.parse_args(argv)
    if args.wheel:
        args.legacy = True
    with runtime_location(args.legacy):
        return install(args)


def install(args) -> int:

    # A failed re-install must never leave an old success marker for callers.
    RUNTIME_JSON.unlink(missing_ok=True)
    if args.allow_missing_psytrainer:
        raise SystemExit("--allow-missing-psytrainer is no longer supported: a complete runtime is required")
    wheel_spec = (args.wheel or "").strip() or None
    wheel_path = resolve_wheel(wheel_spec) if args.legacy else None
    required = wheel_python(wheel_path) if wheel_path else None
    wheelhouse = args.wheelhouse
    if wheelhouse is None and (ROOT / "wheelhouse").is_dir():
        wheelhouse = ROOT / "wheelhouse"
    if wheelhouse is not None and not wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse directory not found: {wheelhouse}")
    manifest = wheelhouse.parent / "bundle.json" if wheelhouse else None
    if manifest and manifest.is_file():
        bundle = json.loads(manifest.read_text(encoding="utf-8"))
        if args.legacy and bundle["profile"] != "legacy":
            raise SystemExit("this offline bundle contains Pipeline dependencies only; install historical INI dependencies separately")
        bundled_python = tuple(bundle["python"])
        if bundled_python not in SUPPORTED_PYTHONS or (required and required != bundled_python):
            raise SystemExit("offline bundle Python does not match the requested runtime")
        required = bundled_python
    base_cmd = resolve_base_python(args.python, required)
    py = ensure_venv(base_cmd, recreate=args.recreate)
    install_requirements(py, wheel_path, wheelhouse)
    status = verify(py, legacy=args.legacy)
    write_runtime(py, status, str(wheel_path.resolve()) if wheel_path else None, base_cmd)
    print("psytrainer-ml runtime install complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
