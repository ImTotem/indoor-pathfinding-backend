from __future__ import annotations

from pathlib import Path


async def build_database(*, session_path: str, output_path: str, **_: object) -> str:
    """Compatibility shim for the old RTABMapEngine process path.

    The current integration uses uploaded `rtabmap.db` files as the source of truth.
    This shim only preserves import compatibility for the legacy queue path.
    """

    source = Path(session_path) / "rtabmap.db"
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"rtabmap.db not found under {session_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return str(target)
