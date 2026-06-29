from __future__ import annotations

from pathlib import Path


def test_windows_controller_uses_portable_repo_output_and_m5_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    controller = root / "Amazon08_run_verified_v4.ps1"
    text = controller.read_text(encoding="utf-8")

    assert "[string]$RepoRoot = $PSScriptRoot" in text
    assert "[string]$OutputRoot = ''" in text
    assert "[string]$M5Input = $env:AMAZON08_M5_INPUT" in text
    assert "$Repo = (Resolve-Path -LiteralPath $RepoRoot).Path" in text
    assert "Provide -M5Input <M5 ZIP or extracted dataset directory>" in text
    assert "C:\\Users\\" not in text
