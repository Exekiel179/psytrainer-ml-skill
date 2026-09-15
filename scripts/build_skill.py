#!/usr/bin/env python3
"""Build the distributable Skill ZIP from a committed Git revision."""

import argparse
import io
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = (
    "README.md", "README.zh-CN.md", "SKILL.md", "NOTICE.md", "LICENSE",
    "requirements.txt", "scripts", "config", "fixtures", "references", "agents", "tests",
)
REQUIRED = {
    "SKILL.md", "requirements.txt", "scripts/install_runtime.py",
    "scripts/pipeline_train.py", "scripts/pipeline_options.py", "tests/smoke_pipeline.py",
}


def build(root, ref, output):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, text=True).strip()
    data = subprocess.check_output(
        ["git", "archive", "--format=zip", "--prefix=psytrainer-ml/", commit, "--", *CONTENTS], cwd=root)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None, "corrupt Skill archive"
        assert archive.comment.decode() == commit, "archive revision mismatch"
        names = {name.removeprefix("psytrainer-ml/") for name in archive.namelist()}
        if not REQUIRED <= names:
            raise ValueError(f"missing Skill files: {sorted(REQUIRED - names)}")
        for item in archive.infolist():
            path = Path(item.filename)
            if (path.is_absolute() or ".." in path.parts
                    or any(part in {".git", ".venv", ".venv-legacy", "__pycache__", "vendor", "wheelhouse"}
                           for part in path.parts)
                    or path.name in {"runtime.json", "runtime-legacy.json", "ml.ini"}
                    or path.suffix in {".whl", ".pyc"}
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"unexpected distributable file: {item.filename}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    return commit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Committed revision to package")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/psytrainer-ml-skill.zip")
    args = parser.parse_args()
    print(f"Packaged commit {build(ROOT, args.ref, args.output)}: {args.output}")
