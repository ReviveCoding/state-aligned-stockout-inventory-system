from __future__ import annotations

from pathlib import Path


def test_repository_wheel_verifier_uses_default_build_isolation() -> None:
    root = Path(__file__).resolve().parents[2]
    verifier = root / "scripts" / "verify_repository.py"
    text = verifier.read_text(encoding="utf-8")

    assert '"--no-build-isolation"' not in text
    assert '"--wheel-dir", str(dist)' in text
    assert "Keep pip build isolation enabled" in text