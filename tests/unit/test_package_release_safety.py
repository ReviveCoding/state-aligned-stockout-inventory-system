from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_package_release():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "package_release.py"
    spec = importlib.util.spec_from_file_location("package_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_package_excludes_rollback_and_generated_tree_artifacts() -> None:
    package_release = _load_package_release()

    assert not package_release.include(
        Path("Amazon08_run_verified_v4.ps1.before-native-stderr-fix.bak")
    )
    assert not package_release.include(
        Path("repo_tree.txt")
    )
    assert package_release.include(
        Path("src/inventory_ai/pipeline.py")
    )


def test_release_package_excludes_local_run_outputs() -> None:
    package_release = _load_package_release()

    assert not package_release.include(
        Path(".local-run/ci_build_isolation/smoke.yaml")
    )
