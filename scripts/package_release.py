from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile

try:
    from _archive_safety import safe_extract_zip
except ModuleNotFoundError:  # imported as scripts.<module> during tests
    from scripts._archive_safety import safe_extract_zip
from pathlib import Path

EXCLUDED_PARTS = {
    ".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", ".ipynb_checkpoints", ".tox", ".nox",
    "dist", "build", ".venv", "venv",
}
EXCLUDED_FILENAMES = {
    ".coverage", ".env", ".DS_Store", "Thumbs.db", "repo_tree.txt",
    # These are release-sidecar artifacts. Embedding final archive hashes
    # inside the archive creates unstable self-referential checksums.
    "qualification_manifest.json", "release_bundle_manifest.json",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak"}
PACKAGE_MANIFEST_PATH = Path("artifacts/package_manifest.json")


def include(path: Path) -> bool:
    if (
        any(
            part in EXCLUDED_PARTS
            or part.endswith(".egg-info")
            or (part.startswith(".") and "-backup-" in part)
            for part in path.parts
        )
        or path.name in EXCLUDED_FILENAMES
        or path.name.startswith(".coverage.")
        or path.suffix in EXCLUDED_SUFFIXES
    ):
        return False
    # Processed datasets and run-specific manifests are reproducible and should
    # not be shipped in a source release. The package manifest is regenerated.
    if len(path.parts) >= 2 and path.parts[:2] == ("data", "processed") and path.name != ".gitkeep":
        return False
    if path.parts and path.parts[0] == "artifacts" and path != PACKAGE_MANIFEST_PATH and path.name != ".gitkeep":
        return False
    # Keep compact decision evidence, but exclude heavy reproducible CSV paths.
    if path.parts and path.parts[0] == "reports" and path.suffix == ".csv":
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else root.parent / f"{root.name}.zip"
    manifest = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and include(relative) and relative != PACKAGE_MANIFEST_PATH:
            manifest.append({"path": relative.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest_path = root / PACKAGE_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"manifest_version": 1, "self_excluded": True, "files": manifest}, indent=2),
        encoding="utf-8",
    )
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and include(relative):
                archive.write(path, Path(root.name) / relative)
    with tempfile.TemporaryDirectory() as temp_dir:
        extracted = safe_extract_zip(output, temp_dir)
        if extracted.name != root.name:
            raise SystemExit(
                f"packaged archive root mismatch: expected {root.name}, got {extracted.name}"
            )
        packaged_manifest = json.loads((extracted / PACKAGE_MANIFEST_PATH).read_text(encoding="utf-8"))
        expected = {entry["path"]: entry for entry in packaged_manifest["files"]}
        actual = {
            path.relative_to(extracted).as_posix(): path
            for path in extracted.rglob("*") if path.is_file()
        }
        allowed = set(expected) | {PACKAGE_MANIFEST_PATH.as_posix()}
        if set(actual) != allowed:
            missing_files = sorted(allowed - set(actual))
            unexpected_files = sorted(set(actual) - allowed)
            raise SystemExit(
                f"package manifest mismatch: missing={missing_files}, unexpected={unexpected_files}"
            )
        for relative, metadata in expected.items():
            path = actual[relative]
            if sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
                raise SystemExit(f"package integrity mismatch: {relative}")
        required = [
            "pyproject.toml", "src/inventory_ai/pipeline.py", "configs/smoke.yaml",
            "scripts/run_pipeline.py", "scripts/run_sql_marts.py", "tests/integration/test_pipeline.py",
        ]
        missing = [item for item in required if not (extracted / item).exists()]
        if missing:
            raise SystemExit(f"packaged archive missing required paths: {missing}")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "sha256": sha256(output), "files": len(manifest), "sidecar": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
