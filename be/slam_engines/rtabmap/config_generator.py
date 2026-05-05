from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigGenerator:
    """Minimal RTAB-Map config writer kept for the legacy `/api/slam/process` path."""

    def generate_config(
        self,
        session_path: str,
        output_dir: str,
        slam_params: dict[str, Any] | None = None,
    ) -> str:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "rtabmap.ini"
        params = slam_params or {}
        lines = [f"{key}={value}" for key, value in sorted(params.items())]
        lines.append(f"DataPath={session_path}")
        config_path.write_text("\n".join(lines) + "\n")
        return str(config_path)
