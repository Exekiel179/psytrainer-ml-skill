import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import install_runtime
import runtime_dependencies as deps
import model_registry


class DependencyTests(unittest.TestCase):
    def setUp(self):
        deps._checked.clear()

    def tearDown(self):
        deps._checked.clear()

    def test_default_setup_creates_environment_without_downloading(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(install_runtime, "RUNTIME_JSON", Path(temp) / "runtime.json"), \
             patch.object(install_runtime, "resolve_base_python", return_value=[sys.executable]), \
             patch.object(install_runtime, "ensure_venv", return_value=Path(sys.executable)), \
             patch.object(install_runtime, "install_requirements") as install, \
             patch.object(install_runtime, "verify") as verify:
            install_runtime.main([])
            install.assert_not_called()
            verify.assert_not_called()
            state = json.loads((Path(temp) / "runtime.json").read_text())
            self.assertTrue(state["ready"])
            self.assertFalse(state["capabilities"]["pipeline"])
            self.assertEqual(state["dependencyMode"], "on-demand")

    def test_scoped_install_passes_only_selected_requirements(self):
        with patch.object(deps, "probe", side_effect=[["numpy"], []]), \
             patch.object(install_runtime, "run") as run:
            install_runtime.install_requirements(Path(sys.executable), specs=deps.requirements(["numpy"]))
        command = run.call_args_list[0].args[0]
        self.assertIn("numpy>=1.26,<3", command)
        self.assertNotIn("-r", command)
        self.assertFalse(any("catboost" in arg for arg in command))

    def test_probe_rechecks_current_versions_offline(self):
        self.assertEqual(deps.probe(sys.executable, ["pip>=1"]), [])
        self.assertEqual(deps.probe(sys.executable, ["pip>9999"]), ["pip>9999"])

    def test_task_selection_covers_models_and_search_resampling(self):
        tags = ["ModelLRRegressor"]
        basic = deps.training_packages(tags, "regression", "none", {tags[0]: [{}]})
        self.assertNotIn("catboost", basic)
        self.assertNotIn("imbalanced-learn", basic)
        self.assertIn("python-docx", basic)
        names = deps.training_packages(["ModelXGBoostClassifier"], "classification", "none",
                                      {"ModelXGBoostClassifier": [{"preprocess__resample": "smote"}]})
        self.assertIn("xgboost", names)
        self.assertIn("imbalanced-learn", names)
        self.assertNotIn("catboost", names)

    def test_available_task_never_installs(self):
        with patch.object(deps, "probe", return_value=[]), \
             patch.object(deps.subprocess, "run"), \
             patch.object(install_runtime, "install_requirements") as install:
            deps.ensure(["numpy"])
        install.assert_not_called()

    def test_successful_repair_restarts_cli_before_using_loaded_libraries(self):
        with patch.object(deps, "probe", side_effect=[["xgboost"], []]), \
             patch.object(deps.subprocess, "run"), \
             patch.object(deps.subprocess, "call", return_value=0) as restart, \
             patch.object(install_runtime, "install_requirements"), \
             patch.object(sys, "argv", ["scripts/pipeline_train.py", "train"]), \
             patch.dict(sys.modules, {"numpy": object()}):
            with self.assertRaises(SystemExit) as result:
                deps.ensure(["xgboost"])
        self.assertEqual(result.exception.code, 0)
        restart.assert_called_once_with([sys.executable, "scripts/pipeline_train.py", "train"])

    def test_failed_download_gives_manual_and_agent_command(self):
        with patch.object(deps, "probe", return_value=["xgboost"]), \
             patch.object(install_runtime, "install_requirements", side_effect=SystemExit("index unreachable")):
            with self.assertRaises(RuntimeError) as error:
                deps.ensure(["xgboost"])
        message = str(error.exception)
        for expected in (sys.executable, "--packages", "xgboost", "coding agent", "--index-url", "--wheelhouse"):
            self.assertIn(expected, message)
        self.assertNotIn("xgboost", deps._checked)

    def test_missing_optional_versions_do_not_break_training(self):
        original = model_registry.metadata.version
        def version(name):
            if name == "catboost":
                raise model_registry.metadata.PackageNotFoundError(name)
            return original(name)
        with patch.object(model_registry.metadata, "version", side_effect=version):
            self.assertIsNone(model_registry.versions()["catboost"])

    def test_prediction_repairs_only_allowlisted_missing_module(self):
        import joblib
        missing = ModuleNotFoundError("missing", name="xgboost.core")
        with patch.object(joblib, "load", side_effect=[missing, {"pipeline": "model"}]), \
             patch.object(deps, "ensure") as ensure:
            self.assertEqual(deps.load_model("model.joblib"), {"pipeline": "model"})
        ensure.assert_called_once_with(["xgboost"])
        with patch.object(joblib, "load", side_effect=ModuleNotFoundError("missing", name="arbitrary_package")), \
             patch.object(deps, "ensure") as ensure:
            with self.assertRaises(ModuleNotFoundError):
                deps.load_model("model.joblib")
        ensure.assert_not_called()
