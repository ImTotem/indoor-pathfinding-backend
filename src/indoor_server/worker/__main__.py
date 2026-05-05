"""python -m indoor_server.worker 진입점."""
from __future__ import annotations

import asyncio
import logging

from indoor_server.config import settings
from indoor_server.worker.loop import WorkerLoop

logging.basicConfig(level=settings.log_level)


def main() -> None:
    asyncio.run(WorkerLoop().run_forever())


if __name__ == "__main__":
    main()
