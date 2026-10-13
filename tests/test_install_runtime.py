from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

def load_installer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "install_runtime.py"
    spec = importlib.util.spec_from_file_location("install_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


install_runtime = load_installer()


class InstallRuntimeTests(unittest.TestCase):
    def test_resolve_wheel_rejects_empty_path(self):
        with self.assertRaisesRegex(SystemExit, "PSYTRAINER_WHEEL|optional-wheel"):
            install_runtime.resolve_wheel(None)

    def test_optional_wheel_discovery_skips_when_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PSYTRAINER_WHEEL", None)
                self.assertIsNone(install_runtime.discover_optional_wheel())
            wheel = root / "PsyTrainer-0.2.0-cp313-none-any.whl"
            wheel.write_bytes(b"wheel")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PSYTRAINER_WHEEL", None)
                self.assertEqual(install_runtime.discover_optional_wheel(), wheel.resolve())
            (root / "PsyTrainer-0.1.0-cp313-none-any.whl").write_bytes(b"other")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PSYTRAINER_WHEEL", None)
                self.assertIsNone(install_runtime.discover_optional_wheel())

    def test_optional_wheel_config_file_is_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / "external.whl"
            wheel.write_bytes(b"wheel")
            config = root / "config" / "optional-wheel.path"
            config.parent.mkdir()
            config.write_text(str(wheel) + "\n", encoding="utf-8")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", config), \
                 patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PSYTRAINER_WHEEL", None)
                self.assertEqual(install_runtime.discover_optional_wheel(), wheel.resolve())

    def test_setup_installs_legacy_when_optional_wheel_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = root / "PsyTrainer-0.2.0-cp313-none-any.whl"
            wheel.write_bytes(b"wheel")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=wheel), \
                 patch.object(install_runtime, "wheel_python", return_value=(3, 13)), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", side_effect=lambda *a, **k: (
                     root / (".venv-legacy" if install_runtime.RUNTIME_JSON.name == "runtime-legacy.json"
                             else ".venv") / "bin" / "python")), \
                 patch.object(install_runtime, "install_requirements") as install, \
                 patch.object(install_runtime, "verify", side_effect=lambda *a, **k: (
                     {"ccpl_training_models": True} if k.get("legacy") else {"sklearn": True})):
                self.assertEqual(install_runtime.main([]), 0)
            self.assertTrue((root / "runtime.json").is_file())
            self.assertTrue((root / "runtime-legacy.json").is_file())
            self.assertEqual(json.loads((root / "runtime-legacy.json").read_text())["profile"], "legacy")
            # Pipeline with no --packages/--all skips installs; optional legacy installs deps + wheel.
            self.assertEqual(install.call_count, 1)
            self.assertEqual(install.call_args.args[1], wheel)

    def test_external_wheel_metadata_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            wheel = Path(temp) / "external.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("PsyTrainer-0.2.0.dist-info/METADATA", "Name: PsyTrainer\nVersion: 0.2.0\n")
                archive.writestr("PsyTrainer-0.2.0.dist-info/WHEEL", "Tag: cp314-none-any\n")
            self.assertEqual(install_runtime.wheel_python(wheel), (3, 14))
            wheel.write_bytes(b"truncated")
            with self.assertRaisesRegex(SystemExit, "invalid PsyTrainer wheel"):
                install_runtime.wheel_python(wheel)

    def test_allow_missing_fails_and_removes_stale_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            runtime.write_text('{"ready": true}')
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None):
                with self.assertRaisesRegex(SystemExit, "no longer supported"):
                    install_runtime.main(["--allow-missing-psytrainer"])
            self.assertFalse(runtime.exists())

    def test_check_only_never_installs_or_creates_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            py = Path(sys.executable)
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "venv_python", return_value=py), \
                 patch.object(install_runtime, "python_version", return_value=(3, 12)), \
                 patch.object(install_runtime, "ensure_venv", side_effect=AssertionError("venv")), \
                 patch.object(install_runtime, "install_requirements", side_effect=AssertionError("install")), \
                 patch("runtime_dependencies.probe", return_value=[]), \
                 patch.object(install_runtime, "verify", return_value={"sklearn": True}):
                self.assertEqual(install_runtime.main(["--check", "--packages", "scikit-learn"]), 0)
            self.assertTrue(runtime.is_file())
            self.assertIsNone(json.loads(runtime.read_text())["wheel"])

    def test_failed_check_removes_stale_ready_marker_without_installing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            runtime.write_text('{"ready": true}')
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "venv_python", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "python_version", return_value=(3, 12)), \
                 patch.object(install_runtime, "install_requirements", side_effect=AssertionError("install")), \
                 patch("runtime_dependencies.probe", return_value=["xgboost>=2,<4"]):
                with self.assertRaises(SystemExit):
                    install_runtime.main(["--check", "--packages", "xgboost"])
            self.assertFalse(runtime.exists())

    def test_failed_install_never_publishes_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements", side_effect=SystemExit("network")):
                with self.assertRaises(SystemExit):
                    install_runtime.main(["--packages", "xgboost"])
            self.assertFalse(runtime.exists())

    def test_failed_training_probe_never_publishes_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", side_effect=SystemExit("broken")):
                with self.assertRaises(SystemExit):
                    install_runtime.main(["--all"])
            self.assertFalse(runtime.exists())

    def test_satisfied_environment_skips_network_installation(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=True), \
             patch.object(install_runtime, "run") as run, \
             patch("runtime_dependencies.probe", return_value=[]):
            install_runtime.install_requirements(Path("python"))
            run.assert_called_once()

    def test_offline_install_uses_no_index_and_checks_dependencies(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime, "run") as run, \
             patch("runtime_dependencies.probe", side_effect=[[], []]):
            install_runtime.install_requirements(Path("python"), wheelhouse=Path("wheels"))
            command = run.call_args_list[0].args[0]
            self.assertIn("--no-index", command)
            self.assertIn("--find-links", command)

    def test_missing_dependencies_use_configured_network_budget(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime, "run") as run, \
             patch("runtime_dependencies.probe", side_effect=[["xgboost>=2,<4"], []]):
            install_runtime.install_requirements(
                Path("python"), index_url="https://example.invalid/simple",
                timeout=11, retries=3, max_seconds=90, specs=["xgboost>=2,<4"])
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--timeout") + 1], "11")
            self.assertEqual(command[command.index("--retries") + 1], "3")
            self.assertEqual(run.call_args.kwargs["timeout"], 90)

    def test_timeout_preserves_environment_and_explains_retry(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime, "run", side_effect=subprocess.TimeoutExpired(cmd=["pip"], timeout=1)), \
             patch("runtime_dependencies.probe", return_value=["xgboost>=2,<4"]):
            with self.assertRaisesRegex(SystemExit, "Rerun the same command"):
                install_runtime.install_requirements(Path("python"), specs=["xgboost>=2,<4"])

    def test_offline_probe_ignores_network_and_upgrade_settings(self):
        captured = {}

        def fake_run(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(args[0], 0, "", "")

        with patch.dict(os.environ, {"PIP_FIND_LINKS": "https://example.invalid/wheels",
                                     "PIP_INDEX_URL": "https://example.invalid/simple",
                                     "PIP_UPGRADE": "1"}), \
             patch("subprocess.run", side_effect=fake_run):
            self.assertTrue(install_runtime.requirements_satisfied(Path(sys.executable)))
        self.assertEqual(captured["env"]["PIP_CONFIG_FILE"], os.devnull)
        self.assertNotIn("PIP_FIND_LINKS", captured["env"])
        self.assertNotIn("PIP_INDEX_URL", captured["env"])
        self.assertNotIn("PIP_UPGRADE", captured["env"])

    def test_uv_targets_venv_and_honors_network_budget(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime.shutil, "which", return_value="/usr/bin/uv"), \
             patch.object(install_runtime, "run") as run, \
             patch("runtime_dependencies.probe", side_effect=[["xgboost>=2,<4"], []]):
            install_runtime.install_requirements(
                Path("python"), index_url="https://example.invalid/simple",
                timeout=9, retries=4, max_seconds=77, installer="uv", specs=["xgboost>=2,<4"])
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["/usr/bin/uv", "pip", "install", "--python"])
            self.assertEqual(run.call_args.kwargs["env"]["UV_HTTP_TIMEOUT"], "9")
            self.assertEqual(run.call_args.kwargs["env"]["UV_HTTP_RETRIES"], "4")
            self.assertEqual(run.call_args.kwargs["timeout"], 77)

    def test_uv_offline_install_cannot_access_index(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime.shutil, "which", return_value="/usr/bin/uv"), \
             patch.object(install_runtime, "run") as run, \
             patch("runtime_dependencies.probe", side_effect=[[], []]):
            install_runtime.install_requirements(Path("python"), wheelhouse=Path("wheels"), installer="uv")
            command = run.call_args_list[0].args[0]
            self.assertIn("--offline", command)
            self.assertIn("--no-index", command)

    def test_missing_uv_does_not_trigger_a_download(self):
        with patch.object(install_runtime, "requirements_satisfied", return_value=False), \
             patch.object(install_runtime.shutil, "which", return_value=None), \
             patch.object(install_runtime, "run", side_effect=AssertionError("run")), \
             patch("runtime_dependencies.probe", return_value=["xgboost>=2,<4"]):
            with self.assertRaisesRegex(SystemExit, "uv is not installed"):
                install_runtime.install_requirements(Path("python"), installer="uv", specs=["xgboost>=2,<4"])

    def test_verification_imports_real_trainer_not_just_package_spec(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "ccpl_training_models"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "trainer.py").write_text("raise ImportError('broken dependency')\n")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.dict(os.environ, {"PYTHONPATH": str(root)}):
                with self.assertRaises(subprocess.CalledProcessError):
                    install_runtime.verify(Path(sys.executable), legacy=True)

    def test_requirement_probe_is_offline_and_respects_version_constraints(self):
        from runtime_dependencies import probe
        self.assertEqual(probe(Path(sys.executable), ["pip"]), [])
        missing = probe(Path(sys.executable), ["pip==0.0.0"])
        self.assertTrue(any(item.startswith("pip==") for item in missing))

    def test_wrong_explicit_python_rejected(self):
        with patch.object(install_runtime, "python_version", return_value=(3, 11)):
            with self.assertRaisesRegex(SystemExit, "3.12-3.14"):
                install_runtime.resolve_base_python("python")

    def test_wrong_existing_venv_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            venv = root / ".venv"
            venv.mkdir()
            with patch.object(install_runtime, "VENV_DIR", venv), \
                 patch.object(install_runtime, "venv_python", return_value=venv / "bin" / "python"), \
                 patch.object(install_runtime, "python_version", side_effect=[(3, 12), (3, 13)]):
                with self.assertRaisesRegex(SystemExit, "different Python"):
                    install_runtime.ensure_venv([sys.executable], recreate=False)

    def test_write_runtime_preserves_venv_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            py = root / "python"
            py.write_text("")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "RUNTIME_JSON", runtime), \
                 patch.object(install_runtime, "VENV_DIR", root / ".venv"), \
                 patch.object(install_runtime, "venv_pip", return_value=root / "pip"):
                install_runtime.write_runtime(py, {"sklearn": True}, None, [str(py)])
            self.assertEqual(json.loads(runtime.read_text())["python"], str(py.absolute()))

    def test_complete_install_publishes_ready_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = root / "runtime.json"
            status = {"pandas": True, "numpy": True, "scipy": True, "joblib": True, "sklearn": True}
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", return_value=status):
                self.assertEqual(install_runtime.main(["--all"]), 0)
            payload = json.loads(runtime.read_text())
            self.assertTrue(payload["ready"])
            self.assertTrue(payload["imports"]["sklearn"])
            self.assertIsNone(payload["wheel"])
            self.assertEqual(payload["capabilities"], {"pipeline": True, "legacy": False})

    def test_pipeline_install_never_resolves_a_vendor_wheel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"PSYTRAINER_WHEEL": "stale/external.whl"}), \
                 patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=None), \
                 patch.object(install_runtime, "resolve_wheel", side_effect=AssertionError("vendor required")), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
                 patch.object(install_runtime, "install_requirements") as install, \
                 patch.object(install_runtime, "verify", return_value={"sklearn": True}):
                self.assertEqual(install_runtime.main(["--all"]), 0)
                self.assertIsNone(install.call_args.args[1])

    def test_supported_pipeline_python_versions(self):
        for version in ((3, 12), (3, 13), (3, 14)):
            with patch.object(install_runtime, "python_version", return_value=version):
                self.assertTrue(install_runtime.resolve_base_python("python"))
        for version in ((3, 11), (3, 15), None):
            with patch.object(install_runtime, "python_version", return_value=version):
                with self.assertRaisesRegex(SystemExit, "3.12-3.14"):
                    install_runtime.resolve_base_python("python")

    def test_legacy_environment_is_isolated_and_profile_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "runtime.json"
            marker.write_text('{"ready": true}')
            wheel = root / "external.whl"
            wheel.write_bytes(b"wheel")
            with patch.object(install_runtime, "ROOT", root), \
                 patch.object(install_runtime, "OPTIONAL_WHEEL_CONFIG", root / "config" / "optional-wheel.path"), \
                 patch.object(install_runtime, "discover_optional_wheel", return_value=wheel), \
                 patch.object(install_runtime, "wheel_python", return_value=(3, 14)), \
                 patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
                 patch.object(install_runtime, "ensure_venv", side_effect=lambda *a, **k: (
                     root / (".venv-legacy" if install_runtime.RUNTIME_JSON.name == "runtime-legacy.json"
                             else ".venv") / "bin" / "python")), \
                 patch.object(install_runtime, "install_requirements"), \
                 patch.object(install_runtime, "verify", side_effect=lambda *a, **k: (
                     {"ccpl_training_models": True} if k.get("legacy") else {"sklearn": True})):
                self.assertEqual(install_runtime.main([]), 0)
            payload = json.loads((root / "runtime-legacy.json").read_text())
            self.assertEqual(payload["profile"], "legacy")
            self.assertTrue(payload["capabilities"]["legacy"])
            # Pipeline marker is rewritten by the first install pass.
            self.assertTrue((root / "runtime.json").is_file())
            self.assertEqual(json.loads((root / "runtime.json").read_text())["profile"], "pipeline")

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
