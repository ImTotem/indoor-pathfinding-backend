import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from storage.postgres_adapter import SCAN_STATUS_PROCESSING, SCAN_STATUS_FAILED

logger = logging.getLogger(__name__)


class SLAMJobQueue:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, postgres_adapter, slam_engine, maps_dir: Path):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.adapter = postgres_adapter
        self.engine = slam_engine
        self.maps_dir = maps_dir
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
        if self.worker_task is not None:
            raise RuntimeError("Worker already running")
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("SLAM job queue worker started")
    
    async def _worker_loop(self):
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
                        
                        result = await self.engine.process(
                            session_id=session_id,
                            frames_data={"input_db_path": input_db_path},
                            progress_callback=None
                        )
                        last_result = result
                        
                        total_nodes = result.get('total_nodes', 0)
                        total_distance = result.get('total_distance', 0.0)
                        
                        await self.adapter.update_processing_result(
                            session_id=session_id,
                            total_nodes=total_nodes,
                            total_distance=total_distance,
                        )
                        logger.info(f"Session processed: session_id={session_id}")
                    
                    if last_result and 'binary' in last_result:
                        self.maps_dir.mkdir(parents=True, exist_ok=True)
                        output_path = self.maps_dir / f"{building_id}.db"
                        output_path.write_bytes(last_result['binary'])
                        logger.info(f"Building map saved: {output_path} ({len(last_result['binary'])} bytes)")
                    
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
