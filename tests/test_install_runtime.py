from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_installer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "install_runtime.py"
    spec = importlib.util.spec_from_file_location("install_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


install_runtime = load_installer()


class InstallRuntimeTests(unittest.TestCase):
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
