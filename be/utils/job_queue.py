import asyncio
import functools
import logging
import math
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import settings
from storage.postgres_adapter import SCAN_STATUS_PROCESSING, SCAN_STATUS_COMPLETED, SCAN_STATUS_FAILED

logger = logging.getLogger(__name__)


def _compute_trajectory_distance(keyframes: list) -> float:
    total = 0.0
    for i in range(1, len(keyframes)):
        prev = keyframes[i - 1]['position']
        curr = keyframes[i]['position']
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        dz = curr[2] - prev[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


class SLAMJobQueue:
    """Async job queue for SLAM processing.

    NOT a singleton — lifespan manages the single instance.
    Previous singleton pattern caused worker stalls on lifespan restart
    because __init__ was skipped and the dead worker_task was reused.
    """

    def __init__(self, postgres_adapter, slam_engine, maps_dir: Path):
        self.adapter = postgres_adapter
        self.engine = slam_engine
        self.maps_dir = maps_dir
        self.tmp_dir = settings.DATA_DIR / "tmp"
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self._shutdown_flag = False

        logger.info("SLAMJobQueue initialized")
    
    async def enqueue(self, building_id: str, session_db_pairs: List[Tuple[str, str]]):
        if self._shutdown_flag:
            raise RuntimeError("Queue is shutting down")
        
        await self.queue.put((building_id, session_db_pairs))
        logger.info(f"Job enqueued: building_id={building_id}, sessions={len(session_db_pairs)}, queue_length={self.queue.qsize()}")
    
    async def start_worker(self):
        if self.worker_task is not None and not self.worker_task.done():
            raise RuntimeError("Worker already running")
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("SLAM job queue worker started")
    
    async def _process_session(self, session_id: str, input_db_path: str) -> dict:
        """Parse an already-processed RTABMap .db uploaded via Spring Boot.

        The uploaded .db is a complete RTABMap database (nodes with poses,
        odometry/loop-closure links, images, depth, calibration) produced by
        the custom RTABMap app on the mobile device.  Because it is already
        fully processed, we only need to parse metadata and copy the map —
        running rtabmap-reprocess would *break* the data (all frames become
        lost=true since -odom tries to recompute odometry from scratch).

        The file is copied to DATA_DIR/tmp/ so that the rtabmap container
        can reach it if we ever need reprocessing in the future.

        TODO: If the uploaded .db ever changes to raw data (images + depth
              only, no odometry links), re-enable _run_reprocess here:
              >>> await self.engine._run_reprocess(
              ...     str(tmp_input), str(tmp_output), progress_callback=None
              ... )
              >>> parsed = await self.engine.database_parser.parse_database(str(tmp_output))
              >>> map_binary = await self.engine._load_map_file(str(tmp_output))
              Also note: _run_reprocess has a pipe deadlock bug where
              process.wait() hangs because stderr is never drained concurrently.
              If re-enabling, replace the engine call with a direct subprocess
              using process.communicate() to avoid the hang.
        """
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        src = Path(input_db_path)
        if not src.exists():
            raise FileNotFoundError(f"Input DB not found: {input_db_path}")

        tmp_input = self.tmp_dir / f"{session_id}_input.db"
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                None, functools.partial(shutil.copy2, str(src), str(tmp_input))
            )
            logger.info(f"Copied input DB to {tmp_input} ({src.stat().st_size} bytes)")

            parsed = await self.engine.database_parser.parse_database(str(tmp_input))
            total_nodes = parsed.get('num_keyframes', 0)
            total_distance = _compute_trajectory_distance(parsed.get('keyframes', []))

            map_binary = await self.engine._load_map_file(str(tmp_input))

            return {
                'binary': map_binary,
                'total_nodes': total_nodes,
                'total_distance': total_distance,
                'parsed': parsed,
            }
        finally:
            for f in (tmp_input,):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
    
    async def _worker_loop(self):
        logger.info("Worker loop running, waiting for jobs...")
        while True:
            try:
                building_id, session_db_pairs = await self.queue.get()
                logger.info(f"Job dequeued: building_id={building_id}, sessions={len(session_db_pairs)}, remaining={self.queue.qsize()}")
                
                try:
                    for session_id, _ in session_db_pairs:
                        await self.adapter.update_status(session_id, SCAN_STATUS_PROCESSING)
                    
                    last_result = None
                    for session_id, input_db_path in session_db_pairs:
                        logger.info(f"Processing session {session_id}: {input_db_path}")
                        
                        result = await self._process_session(session_id, input_db_path)
                        last_result = result
                        
                        await self.adapter.update_processing_result(
                            session_id=session_id,
                            total_nodes=result['total_nodes'],
                            total_distance=result['total_distance'],
                        )
                        logger.info(f"Session processed: session_id={session_id}, nodes={result['total_nodes']}, distance={result['total_distance']:.2f}")
                    
                    if last_result and last_result.get('binary'):
                        self.maps_dir.mkdir(parents=True, exist_ok=True)
                        output_path = self.maps_dir / f"{building_id}.db"
                        output_path.write_bytes(last_result['binary'])
                        logger.info(f"Building map saved: {output_path} ({len(last_result['binary'])} bytes)")
                    
                    for session_id, _ in session_db_pairs:
                        await self.adapter.update_status(session_id, SCAN_STATUS_COMPLETED)
                    
                    logger.info(f"Building processing succeeded: building_id={building_id}")
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    for session_id, _ in session_db_pairs:
                        await self.adapter.update_status(session_id, SCAN_STATUS_FAILED, error_msg)
                    logger.error(f"Building processing failed: building_id={building_id}, error={error_msg}", exc_info=True)
                
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
    
    def get_queue_length(self) -> int:
        return self.queue.qsize()
    
    async def shutdown(self):
        if self._shutdown_flag:
            return
        self._shutdown_flag = True
        
        if self.worker_task is None:
            return
        
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass
        
        self.worker_task = None
        logger.info("SLAMJobQueue shutdown complete")
