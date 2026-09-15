import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_skill import build, CONTENTS, REQUIRED


class SkillArchiveTests(unittest.TestCase):
    def repository(self, root):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for name in CONTENTS:
            path = root / name
            if "." in name:
                path.write_text("committed\n")
            else:
                path.mkdir()
                (path / "fixture.txt").write_text("fixture\n")
        for name in REQUIRED:
            (root / name).write_text("committed\n")
        self.commit(root)

    def commit(self, root):
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=root, check=True)

    def test_exact_commit_reproducible_and_excludes_local_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            self.repository(root)
            (root / "SKILL.md").write_text("uncommitted\n")
            (root / "scripts/local-secret.txt").write_text("untracked\n")
            first, second = Path(temp) / "first.zip", Path(temp) / "second.zip"
            commit = build(root, "HEAD", first)
            build(root, commit, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.read("psytrainer-ml/SKILL.md"), b"committed\n")
                self.assertNotIn("psytrainer-ml/scripts/local-secret.txt", archive.namelist())
                self.assertEqual(archive.comment.decode(), commit)
            with self.assertRaises(FileExistsError):
                build(root, commit, first)

    def test_incomplete_or_contaminated_commit_is_not_published(self):
        for invalid in ("missing", "wheel", "runtime"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "repo"
                self.repository(root)
                if invalid == "missing":
                    (root / "scripts/pipeline_options.py").unlink()
                else:
                    name = "scripts/PsyTrainer.whl" if invalid == "wheel" else "scripts/runtime.json"
                    (root / name).write_text("invalid\n")
                self.commit(root)
                output = Path(temp) / "skill.zip"
                with self.assertRaises(ValueError):
                    build(root, "HEAD", output)
                self.assertFalse(output.exists())
