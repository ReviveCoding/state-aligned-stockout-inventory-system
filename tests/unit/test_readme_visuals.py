from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path


EXPECTED = {
    "m5_confirmation_scorecard.svg",
    "m5_heldout_outcomes.svg",
    "m5_reliability_checks.svg",
}


def test_readme_visuals_are_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    generator = root / "scripts" / "generate_readme_visuals.py"

    result = subprocess.run(
        [
            sys.executable,
            str(generator),
            "--root",
            str(root),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert {path.name for path in tmp_path.glob("*.svg")} == EXPECTED

    for name in EXPECTED:
        path = tmp_path / name
        assert path.stat().st_size > 1000
        assert element_tree.parse(path).getroot().tag.endswith("svg")
        assert all(byte < 128 for byte in path.read_bytes())


def test_readme_links_generated_visuals() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "<!-- README_VISUALS_START -->" in readme
    assert "<!-- README_VISUALS_END -->" in readme

    for name in EXPECTED:
        assert f"docs/figures/{name}" in readme