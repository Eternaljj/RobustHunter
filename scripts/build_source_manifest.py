from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"SHA256SUMS", "RobustHunter-GraphMatching.zip"}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "outputs",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)
    if (
        not path.is_file()
        or path.name in EXCLUDED
        or any(part in EXCLUDED_DIRECTORIES for part in relative.parts)
    ):
        continue
    rows.append(f"{digest(path)}  {relative.as_posix()}")
(ROOT / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} hashes")
