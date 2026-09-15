"""Task-scoped dependency checks and bounded repair for command-line workflows."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "joblib": "joblib",
    "scikit-learn": "sklearn", "imbalanced-learn": "imblearn",
    "matplotlib": "matplotlib", "python-docx": "docx",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
}
CORE = ("numpy", "pandas", "scipy", "joblib", "scikit-learn")
REPORT = ("matplotlib", "python-docx")
_checked = set()


def requirements(names):
    # The maintained manifest contains one ordinary requirement per line.
    from re import match
    entries = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name = match(r"[A-Za-z0-9_-]+", line).group().lower()
            entries[name] = line
    unknown = set(names) - set(MODULES)
    if unknown:
        raise ValueError(f"unsupported dependency names: {sorted(unknown)}")
    return [entries[name] for name in dict.fromkeys(names)]


def probe(py, specs):
    """Check installed versions and their dependency closure, entirely offline."""
    if not specs:
        return []
    code = """
import json, sys
from importlib import metadata
from pip._vendor.packaging.requirements import Requirement
pending = [Requirement(s) for s in json.loads(sys.argv[1])]
seen, missing = set(), []
while pending:
    req = pending.pop()
    if req.marker and not req.marker.evaluate():
        continue
    if str(req) in seen:
        continue
    seen.add(str(req))
    try:
        dist = metadata.distribution(req.name)
    except metadata.PackageNotFoundError:
        missing.append(str(req))
        continue
    if not req.specifier.contains(dist.version, prereleases=True):
        missing.append(str(req))
        continue
    pending.extend(Requirement(s) for s in (dist.requires or []))
print(json.dumps(sorted(set(missing))))
"""
    result = subprocess.check_output([str(py), "-c", code, json.dumps(specs)], text=True, timeout=60)
    return json.loads(result)


def repair_message(py, names, detail):
    argv = [str(py), str(ROOT / "scripts/install_runtime.py"), "--packages", *names]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    if os.name == "nt":
        command = "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return (f"Task dependencies unavailable: {', '.join(names)}. {detail}\n"
            f"Install manually, or ask your coding agent to execute this command:\n{command}\n"
            "If the package index is unreachable, add --index-url <reachable-index-url>, "
            "or --wheelhouse <local-dependency-directory> for offline installation. "
            "Then rerun the original task. Do not skip these dependencies or report task success.")


def ensure(names):
    names = list(dict.fromkeys(name for name in names if name not in _checked))
    if not names:
        return
    specs = requirements(names)
    try:
        missing = probe(sys.executable, specs)
        if missing:
            import install_runtime
            config = {}
            runtime = ROOT / "runtime.json"
            if runtime.is_file():
                config = json.loads(runtime.read_text()).get("installation", {})
            wheelhouse = config.get("wheelhouse")
            if not wheelhouse and (ROOT / "wheelhouse").is_dir():
                wheelhouse = str(ROOT / "wheelhouse")
            with contextlib.redirect_stdout(sys.stderr):
                print(f"Missing/incompatible dependencies: {', '.join(missing)}", file=sys.stderr)
                install_runtime.install_requirements(
                    Path(sys.executable), wheelhouse=Path(wheelhouse) if wheelhouse else None,
                    specs=specs, **{k: v for k, v in config.items() if k != "wheelhouse"})
            if probe(sys.executable, specs):
                raise RuntimeError("Dependencies still do not satisfy the task after installation")
            importlib.invalidate_caches()
        # Import in a subprocess so a broken native library cannot kill the task process.
        code = "import importlib; " + "; ".join(
            f"importlib.import_module({MODULES[name]!r})" for name in names)
        subprocess.run([sys.executable, "-c", code], check=True, stdout=sys.stderr, timeout=120)
    except (SystemExit, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        raise RuntimeError(repair_message(sys.executable, names, str(exc))) from exc
    # Optional packages may update a core dependency already imported by the CLI.
    if missing and "numpy" in sys.modules and Path(sys.argv[0]).name in {"pipeline_train.py", "ml.py"}:
        raise SystemExit(subprocess.call([sys.executable, *sys.argv]))
    _checked.update(names)


def bootstrap(script, names):
    """Ensure task dependencies before importing scientific libraries in CLI entrypoints."""
    import install_runtime
    py = install_runtime.venv_python()
    if Path(sys.executable).absolute() != py.absolute():
        if not py.is_file():
            with contextlib.redirect_stdout(sys.stderr):
                install_runtime.main([])
        raise SystemExit(subprocess.call([str(py), str(script), *sys.argv[1:]]))
    try:
        ensure(names)
    except RuntimeError as exc:
        print(json.dumps({"status": "dependencies_required", "error": str(exc)}))
        raise SystemExit(2) from exc


def training_packages(tags, task, resample, spaces):
    from model_registry import MODELS
    names = [*CORE, *REPORT]
    names.extend(MODELS[task][tag][0].split(".")[0] for tag in tags
                 if not MODELS[task][tag][0].startswith("sklearn."))
    if resample != "none" or any(
        candidate.get("preprocess__resample", "none") != "none"
        for candidates in spaces.values() for candidate in candidates
    ):
        names.append("imbalanced-learn")
    return list(dict.fromkeys(names))


def load_model(path):
    """Old model files have no dependency manifest; repair only known missing modules."""
    import joblib
    attempted = set()
    while True:
        try:
            saved = joblib.load(path)
            pipeline = saved.get("pipeline") if isinstance(saved, dict) else saved
            objects = [pipeline]
            if hasattr(pipeline, "get_params"):
                objects.extend(pipeline.get_params(deep=True).values())
            modules = {type(item).__module__.split(".")[0] for item in objects}
            names = [name for name, module in MODULES.items() if module in modules]
            if names:
                ensure(names)
            return saved
        except ModuleNotFoundError as exc:
            module = (exc.name or "").split(".")[0]
            name = next((name for name, value in MODULES.items() if value == module), None)
            if not name or name in attempted:
                raise
            attempted.add(name)
            ensure([name])
