from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_installer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "install_runtime.py"
    spec = importlib.util.spec_from_file_location("install_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


install_runtime = load_installer()


class InstallRuntimeTests(unittest.TestCase):
    def test_bundled_wheel_is_complete_and_python314(self):
        wheel = install_runtime.resolve_wheel(None)
        self.assertEqual(install_runtime.wheel_python(wheel), (3, 14))

    def test_missing_or_corrupt_bundle_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            wheel = Path(temp) / "missing.whl"
            with patch.object(install_runtime, "BUNDLED_WHEEL", wheel):
                with self.assertRaisesRegex(SystemExit, "incomplete download"):
                    install_runtime.resolve_wheel(None)
                wheel.write_bytes(b"truncated")
                with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                    install_runtime.resolve_wheel(None)

    def test_allow_missing_fails_and_removes_stale_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime.json"
            runtime.write_text('{"ready": true}')
            with patch.object(install_runtime, "RUNTIME_JSON", runtime):
                with self.assertRaisesRegex(SystemExit, "complete runtime"):
                    install_runtime.main(["--allow-missing-psytrainer"])
            self.assertFalse(runtime.exists())

    def test_write_runtime_preserves_venv_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            py = root / "python"
            try:
                py.symlink_to(sys.executable)
            except OSError:
                self.skipTest("symlinks unavailable")
            runtime = root / "runtime.json"
            with patch.object(install_runtime, "RUNTIME_JSON", runtime):
                install_runtime.write_runtime(py, {"ccpl_training_models": True}, "wheel", [])
            payload = json.loads(runtime.read_text())
            self.assertEqual(payload["python"], str(py))
            self.assertTrue(payload["ready"])

    def test_wrong_explicit_python_rejected(self):
        with patch.object(install_runtime, "python_version", return_value=(3, 13)):
            with self.assertRaisesRegex(SystemExit, "3.14 required"):
                install_runtime.resolve_base_python("python", (3, 14))

    def test_wrong_existing_venv_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(install_runtime, "VENV_DIR", Path(temp)), \
                 patch.object(install_runtime, "python_version", side_effect=[(3, 12), (3, 14)]), \
                 patch.object(install_runtime, "run") as run:
                with self.assertRaisesRegex(SystemExit, "--recreate"):
                    install_runtime.ensure_venv(["python3.14"], recreate=False)
                run.assert_not_called()

    def test_verification_imports_real_trainer_not_just_package_spec(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("pandas", "numpy", "joblib"):
                (root / f"{name}.py").write_text("")
            package = root / "ccpl_training_models"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "trainer.py").write_text("raise ImportError('broken dependency')")
            import os
            with patch.dict(os.environ, {"PYTHONPATH": str(root)}):
                with self.assertRaises(subprocess.CalledProcessError):
                    install_runtime.verify(Path(sys.executable))

    def test_failed_install_never_publishes_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime.json"
            with patch.object(install_runtime, "RUNTIME_JSON", runtime), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements", side_effect=RuntimeError("download failed")):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    install_runtime.main([])
            self.assertFalse(runtime.exists())

    def test_offline_install_uses_no_index_and_checks_dependencies(self):
        with patch.object(install_runtime, "run") as run:
            install_runtime.install_requirements(Path("python"), Path("vendor.whl"), Path("wheels"))
        command = run.call_args_list[0].args[0]
        self.assertIn("--no-index", command)
        self.assertIn("--find-links", command)
        self.assertIn("vendor.whl", command)
        self.assertEqual(run.call_args_list[1].args[0][-2:], ["pip", "check"])

    def test_failed_training_probe_never_publishes_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime.json"
            runtime.write_text('{"ready": true}')
            with patch.object(install_runtime, "RUNTIME_JSON", runtime), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", side_effect=RuntimeError("trainer unavailable")):
                with self.assertRaisesRegex(RuntimeError, "trainer unavailable"):
                    install_runtime.main([])
            self.assertFalse(runtime.exists())

    def test_complete_install_publishes_ready_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "runtime.json"
            status = dict.fromkeys(["pandas", "numpy", "joblib", "ccpl_training_models"], True)
            with patch.object(install_runtime, "RUNTIME_JSON", runtime), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", return_value=status):
                self.assertEqual(install_runtime.main([]), 0)
            payload = json.loads(runtime.read_text())
            self.assertTrue(payload["ready"])
            self.assertTrue(payload["imports"]["ccpl_training_models"])

    def test_resolve_wheel_exact_path(self):
        with tempfile.TemporaryDirectory() as temp:
            wheel = Path(temp) / "PsyTrainer-0.2.0-cp313-none-any.whl"
            wheel.write_bytes(b"wheel")
            resolved = install_runtime.resolve_wheel(str(wheel))
            self.assertEqual(resolved, wheel.resolve())

    def test_resolve_wheel_glob_single_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / "PsyTrainer-0.2.0-cp313-none-any.whl"
            wheel.write_bytes(b"wheel")
            resolved = install_runtime.resolve_wheel(str(root / "PsyTrainer-*.whl"))
            self.assertEqual(resolved, wheel.resolve())

    def test_resolve_wheel_glob_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "PsyTrainer-0.1.0-cp313-none-any.whl").write_bytes(b"a")
            (root / "PsyTrainer-0.2.0-cp313-none-any.whl").write_bytes(b"b")
            with self.assertRaises(SystemExit):
                install_runtime.resolve_wheel(str(root / "PsyTrainer-*.whl"))

    def test_venv_python_path_shape(self):
        name = install_runtime.venv_python().name
        if install_runtime.IS_WINDOWS:
            self.assertEqual(name, "python.exe")
        else:
            self.assertEqual(name, "python")


if __name__ == "__main__":
    unittest.main()
