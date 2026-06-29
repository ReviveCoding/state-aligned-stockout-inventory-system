from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_ai import __version__
from inventory_ai.config import load_config
from inventory_ai.utils.io import sha256_file

REQUIRED_PATHS = [
    "pyproject.toml", "README.md", "LICENSE", "Dockerfile", "Makefile",
    "configs/smoke.yaml", "configs/m5_smoke.yaml", "src/inventory_ai/pipeline.py",
    "scripts/run_pipeline.py", "scripts/run_sql_marts.py", "scripts/run_runtime_smoke.py",
    "scripts/run_chronos.py", "scripts/run_spark_features.py",
    "tests/integration/test_pipeline.py", ".github/workflows/ci.yml",
]


def run(command: list[str], root: Path, *, pythonpath: bool = True, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    if pythonpath:
        env["PYTHONPATH"] = str(root / "src")
    else:
        env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def verify_manifest(root: Path, cfg) -> None:
    manifest_path = root / cfg.artifact_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"run manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_dir = root / cfg.output_dir
    for section in ("output_files", "sql_outputs"):
        entries = manifest.get(section, {})
        if section == "sql_outputs" and not entries:
            raise SystemExit("SQL provenance is missing from run manifest")
        for name, metadata in entries.items():
            path = report_dir / name
            if not path.exists():
                raise SystemExit(f"manifest output is missing: {path}")
            if sha256_file(path) != metadata["sha256"]:
                raise SystemExit(f"manifest hash mismatch: {path}")
            if path.stat().st_size != int(metadata.get("bytes", -1)):
                raise SystemExit(f"manifest size mismatch: {path}")
    sample = manifest.get("processed_sample")
    if not sample:
        raise SystemExit("processed sample metadata is missing from run manifest")
    sample_path = root / sample["path"]
    if not sample_path.exists() or sha256_file(sample_path) != sample["sha256"]:
        raise SystemExit("processed sample hash mismatch")
    if sample_path.stat().st_size != int(sample.get("bytes", -1)):
        raise SystemExit("processed sample size mismatch")


def verify_wheel(root: Path) -> None:
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    for stale in dist.glob("*.whl"):
        stale.unlink()
    # `pip wheel` is available in standard virtual environments and avoids
    # making repository verification depend on the optional `build` package.
    # No build isolation keeps this validation offline and reproducible.
    run([
        sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
        "--no-build-isolation", "--wheel-dir", str(dist),
    ], root)
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced no artifact")
    wheel = wheels[-1]
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit("wheel contains invalid or ambiguous METADATA")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        if f"Version: {__version__}" not in metadata:
            raise SystemExit(
                f"wheel metadata version does not match runtime version {__version__}"
            )
        for dependency in ("numpy", "pandas", "scikit-learn", "PyYAML"):
            if f"Requires-Dist: {dependency}" not in metadata:
                raise SystemExit(f"wheel metadata is missing dependency: {dependency}")
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "site"
        run([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "--target", str(target), str(wheel),
        ], Path(temp_dir), pythonpath=False)
        isolated_env = {"PYTHONPATH": str(target)}
        run(
            [sys.executable, "-c", "import inventory_ai; from inventory_ai.config import PipelineConfig; print(PipelineConfig().run_name)"],
            Path(temp_dir), pythonpath=False, extra_env=isolated_env,
        )
        run(
            [sys.executable, "-m", "inventory_ai.cli", "--version"],
            Path(temp_dir), pythonpath=False, extra_env=isolated_env,
        )
        run(
            [sys.executable, "-m", "inventory_ai.cli", "--help"],
            Path(temp_dir), pythonpath=False, extra_env=isolated_env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify structure, tests, pipeline, SQL, manifests, and wheel installation.")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-iterate", action="store_true", help="Accept a noncritical ITERATE release status.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        raise SystemExit(f"missing required repository paths: {missing}")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)
    for path in sorted((root / "configs").glob("*.yaml")):
        load_config(path)
    run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], root)
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"], root)
    if not args.skip_pipeline:
        run([sys.executable, "scripts/run_pipeline.py", "--config", str(config_path)], root)
        run([sys.executable, "scripts/run_sql_marts.py", "--config", str(config_path)], root)
    gate_path = root / cfg.output_dir / "release_gate.json"
    if not gate_path.exists():
        raise SystemExit(f"release gate is missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["gate_status"] == "FAIL":
        raise SystemExit("release gate returned FAIL")
    if gate["gate_status"] != "PASS" and not args.allow_iterate:
        raise SystemExit(f"release gate must PASS for repository verification; got {gate['gate_status']}")
    for sql_output in ["sql_daily_demand_mart.csv", "sql_inventory_kpi_mart.csv"]:
        path = root / cfg.output_dir / sql_output
        if not path.exists() or path.stat().st_size <= 20:
            raise SystemExit(f"SQL output is missing or empty: {path}")
    verify_manifest(root, cfg)
    if not args.skip_build:
        verify_wheel(root)
    print(json.dumps({"repository_verification": "PASS", "run_name": cfg.run_name, "gate_status": gate["gate_status"]}, indent=2))


if __name__ == "__main__":
    main()
