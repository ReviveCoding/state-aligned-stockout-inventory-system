from __future__ import annotations

import stat
import zipfile
from pathlib import Path, PurePosixPath


def _validated_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if not normalized or member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if member.parts and ":" in member.parts[0]:
        raise ValueError(f"unsafe archive drive-qualified path: {name!r}")
    return member


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> Path:
    """Validate and extract a source archive, returning its single root directory.

    The verifier rejects CRC failures, absolute/traversal paths, symlinks, and
    archives with more than one top-level repository root before writing files.
    """
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"archive CRC check failed: {bad_member}")
        validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        roots: set[str] = set()
        seen_members: set[str] = set()
        for info in archive.infolist():
            member = _validated_member_path(info.filename)
            normalized_name = member.as_posix()
            if normalized_name in seen_members:
                raise ValueError(f"duplicate archive member is not allowed: {info.filename}")
            seen_members.add(normalized_name)
            roots.add(member.parts[0])
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted archive member is not allowed: {info.filename}")
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive symlink is not allowed: {info.filename}")
            # Some ZIP creators store permission bits without POSIX file-type
            # bits. Treat type=0 as an ordinary file, but reject explicit
            # devices, sockets, FIFOs, and other special entries.
            if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"non-regular archive member is not allowed: {info.filename}")
            target = (destination / Path(*member.parts)).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"archive member escapes destination: {info.filename}")
            validated.append((info, member))
        if len(roots) != 1:
            raise ValueError(f"archive must contain exactly one top-level root; found {sorted(roots)}")
        root_name = next(iter(roots))
        for info, member in validated:
            target = destination / Path(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
    root = destination / root_name
    if not root.is_dir():
        raise ValueError("archive top-level root is not a directory")
    return root
