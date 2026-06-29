from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from _archive_safety import safe_extract_zip
except ModuleNotFoundError:  # imported as scripts.<module> during tests
    from scripts._archive_safety import safe_extract_zip

PACKAGE_MANIFEST = Path("artifacts/package_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    root: Path,
    timeout: int = 240,
    *,
    quiet: bool = False,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    print("+", " ".join(command), flush=True)
    if quiet:
        # Use a disk-backed temporary log rather than PIPE. Numerical stages can
        # emit large JSON payloads; a file avoids pipe buffering and lets us
        # surface diagnostics only when the child fails or times out.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    check=False,
                    timeout=timeout,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                log.seek(0)
                diagnostic = log.read()
                if diagnostic:
                    print(diagnostic, file=sys.stderr)
                raise
            if completed.returncode != 0:
                log.seek(0)
                diagnostic = log.read()
                if diagnostic:
                    print(diagnostic, file=sys.stderr)
                raise subprocess.CalledProcessError(completed.returncode, command)
        return
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def verify_static(extracted: Path) -> None:
    manifest_path = extracted / PACKAGE_MANIFEST
    if not manifest_path.exists():
        raise SystemExit("package manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest.get("files", [])}
    actual = {
        path.relative_to(extracted).as_posix(): path
        for path in extracted.rglob("*") if path.is_file()
    }
    allowed = set(expected) | {PACKAGE_MANIFEST.as_posix()}
    if set(actual) != allowed:
        raise SystemExit(
            f"package file-set mismatch: missing={sorted(allowed-set(actual))}, "
            f"unexpected={sorted(set(actual)-allowed)}"
        )
    for relative, metadata in expected.items():
        path = actual[relative]
        if sha256(path) != metadata["sha256"]:
            raise SystemExit(f"package hash mismatch: {relative}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise SystemExit(f"package size mismatch: {relative}")
    required = {
        "pyproject.toml", "src/inventory_ai/pipeline.py", "configs/smoke.yaml",
        "scripts/run_pipeline.py", "scripts/run_sql_marts.py", "scripts/run_runtime_smoke.py",
        "tests/integration/test_pipeline.py", ".github/workflows/ci.yml",
    }
    missing = sorted(required - set(actual))
    if missing:
        raise SystemExit(f"archive missing required paths: {missing}")


def verify_runtime(extracted: Path) -> None:
    # Pipeline and SQL share one fresh child process. This both validates the
    # real stage connection and avoids repeated BLAS interpreter startup in
    # constrained environments.
    run(
        [
            sys.executable,
            "scripts/run_runtime_smoke.py",
            "--config",
            "configs/smoke.yaml",
        ],
        extracted,
        timeout=300,
        quiet=True,
    )
    print("[archive] connected pipeline and SQL smoke complete", flush=True)
    report_dir = extracted / "reports" / "synthetic_smoke"
    artifact_dir = extracted / "artifacts" / "synthetic_smoke"
    gate = json.loads((report_dir / "release_gate.json").read_text(encoding="utf-8"))
    if gate.get("gate_status") != "PASS":
        raise SystemExit(f"extracted runtime gate is not PASS: {gate.get('gate_status')}")
    manifest = json.loads((artifact_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for section in ("output_files", "sql_outputs"):
        entries = manifest.get(section, {})
        if not entries:
            raise SystemExit(f"runtime manifest missing {section}")
        for name, metadata in entries.items():
            path = report_dir / name
            if not path.exists() or sha256(path) != metadata["sha256"]:
                raise SystemExit(f"runtime artifact hash mismatch: {path}")
            if path.stat().st_size != int(metadata["bytes"]):
                raise SystemExit(f"runtime artifact size mismatch: {path}")
    sample = manifest.get("processed_sample", {})
    sample_path = extracted / sample.get("path", "")
    if not sample_path.exists() or sha256(sample_path) != sample.get("sha256"):
        raise SystemExit("runtime processed sample mismatch")
    if sample_path.stat().st_size != int(sample.get("bytes", -1)):
        raise SystemExit("runtime processed sample size mismatch")
    print("[archive] runtime artifacts verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a packaged source archive in a fresh extraction.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--mode", choices=("static", "tests", "runtime"), default="static")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    if not archive.exists():
        raise SystemExit(f"archive does not exist: {archive}")
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if sidecar.exists():
        expected_hash = sidecar.read_text(encoding="utf-8").split()[0]
        if sha256(archive) != expected_hash:
            raise SystemExit("archive checksum sidecar mismatch")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            extracted = safe_extract_zip(archive, temp_dir)
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        verify_static(extracted)
        if args.mode == "tests":
            run([sys.executable, "-m", "pytest", "-q"], extracted)
        elif args.mode == "runtime":
            verify_runtime(extracted)
        print("[archive] extraction checks complete", flush=True)
    print(json.dumps({"archive_verification": "PASS", "mode": args.mode, "archive": str(archive)}, indent=2))


if __name__ == "__main__":
    main()
