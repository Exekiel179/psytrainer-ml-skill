import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_bundle
import install_runtime


class BundleTests(unittest.TestCase):
    def test_pipeline_bundle_records_python_and_packages_local_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "skill.zip"
            with patch.object(sys, "argv", ["build_bundle.py", "--output", str(output)]), \
                 patch.object(build_bundle, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(build_bundle, "python_version", return_value=(3, 12)), \
                 patch.object(build_bundle.subprocess, "run") as run:
                self.assertEqual(build_bundle.main(), 0)
            download = run.call_args_list[1].args[0]
            self.assertNotIn("None", download)
            self.assertFalse(any(arg.endswith(".whl") for arg in download))
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("psytrainer-ml/bundle.json"))
                self.assertEqual(manifest["python"], [3, 12])
                self.assertEqual(manifest["profile"], "pipeline")
                self.assertIn("psytrainer-ml/scripts/model_registry.py", archive.namelist())
                self.assertIn("psytrainer-ml/README.zh-CN.md", archive.namelist())
                self.assertFalse(any("/vendor/" in n or "psytrainer-0." in n.lower() for n in archive.namelist()))
                self.assertIn("psytrainer-ml/scripts/pipeline_options.py", archive.namelist())
                self.assertFalse(any("/.venv" in n or n.endswith("/runtime.json") for n in archive.namelist()))

    def test_failed_download_does_not_publish_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "skill.zip"
            with patch.object(sys, "argv", ["build_bundle.py", "--output", str(output)]), \
                 patch.object(build_bundle, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(build_bundle, "python_version", return_value=(3, 12)), \
                 patch.object(build_bundle.subprocess, "run", side_effect=[None, RuntimeError("download failed")]):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    build_bundle.main()
            self.assertFalse(output.exists())

    def test_installer_uses_bundle_python_and_rejects_wrong_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "wheelhouse").mkdir()
            (root / "bundle.json").write_text(json.dumps({"profile": "pipeline", "python": [3, 12]}))
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "RUNTIME_JSON", root / "runtime.json"), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]) as resolve, \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", return_value={"sklearn": True}):
                self.assertEqual(install_runtime.main([]), 0)
                self.assertEqual(resolve.call_args.args[1], (3, 12))
                with self.assertRaisesRegex(SystemExit, "provide --legacy --wheel"):
                    install_runtime.main(["--legacy"])
                self.assertTrue((root / "runtime.json").is_file())


if __name__ == "__main__":
    unittest.main()
