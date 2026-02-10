"""
SLAM job queue for sequential processing.
Implements singleton pattern with asyncio.Queue for thread-safe FIFO job handling.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SLAMJobQueue:
    """
    Singleton job queue for sequential SLAM processing.
    
    Ensures only one SLAM job is processed at a time to prevent resource
    contention and maintain system stability.
    
    Features:
    - Singleton pattern (only one instance exists)
    - FIFO queue using asyncio.Queue
    - Sequential processing (one job at a time)
    - Graceful shutdown with task cancellation
    - Database status tracking (pending → in_progress → success/failed)
    
    Usage:
        # Initialize with dependencies
        queue = SLAMJobQueue(postgres_adapter, slam_engine)
        
        # Start worker
        await queue.start_worker()
        
        # Enqueue jobs
        await queue.enqueue(map_id, session_id)
        
        # Check queue length
        length = queue.get_queue_length()
        
        # Shutdown gracefully
        await queue.shutdown()
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, postgres_adapter, slam_engine):
        """
        Initialize job queue with dependencies.
        
        Args:
            postgres_adapter: PostgresAdapter instance for database operations
            slam_engine: RTABMapEngine instance for SLAM processing
        
        Note:
            Due to singleton pattern, only the first __init__ call will
            initialize the instance. Subsequent calls are no-ops.
        """
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.adapter = postgres_adapter
        self.engine = slam_engine
        self.queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self._shutdown_flag = False
        
        logger.info("SLAMJobQueue initialized")
    
    async def enqueue(self, map_id: str, session_id: str, frames_data: dict):
        """
        Add a job to the queue.
        
        Args:
            map_id: Map identifier (UUID hex format)
            session_id: Session identifier
            frames_data: Frame data dict with 'session_path' and 'poses'
        
        Raises:
            RuntimeError: If queue is shutting down
        """
        if self._shutdown_flag:
            raise RuntimeError("Queue is shutting down - not accepting new jobs")
        
        # Update status to 'pending' before enqueuing
        await self.adapter.update_job_status(map_id, 'pending', None)
        
        # Add to queue
        await self.queue.put((map_id, session_id, frames_data))
        
        logger.info(
            f"Job enqueued: map_id={map_id}, session_id={session_id}, "
            f"queue_length={self.queue.qsize()}"
        )
    
    async def start_worker(self):
        """
        Start background worker task for processing jobs.
        
        The worker runs in a loop, processing jobs sequentially until
        cancelled or shutdown is requested.
        
        Raises:
            RuntimeError: If worker is already running
        """
        if self.worker_task is not None:
            raise RuntimeError("Worker already running")
        
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("SLAM job queue worker started")
    
    async def _worker_loop(self):
        """
        Background worker that processes jobs sequentially.
        
        Loop behavior:
        1. Block until job available (Queue.get())
        2. Update status to 'in_progress'
        3. Execute SLAM processing
        4. Update status to 'success' or 'failed'
        5. Mark task as done
        6. Repeat
        
        Exits on:
        - asyncio.CancelledError (graceful shutdown)
        - Critical exceptions (logged and re-raised)
        """
        logger.info("Worker loop started")
        
        while True:
            try:
                # Block until job available
                map_id, session_id, frames_data = await self.queue.get()
                logger.info(
                    f"Job dequeued: map_id={map_id}, session_id={session_id}, "
                    f"remaining_jobs={self.queue.qsize()}"
                )
                
                # Update status to in_progress
                try:
                    await self.adapter.update_job_status(map_id, 'in_progress', None)
                    logger.info(f"Processing started: map_id={map_id}")
                    
                    # Call SLAM engine (no progress callback for async processing)
                    result = await self.engine.process(
                        session_id=session_id,
                        frames_data=frames_data,
                        progress_callback=None
                    )
                    
                    # Store output database
                    if 'binary' in result:
                        await self.adapter.store_output_db(map_id, result['binary'])
                        logger.info(
                            f"Output database stored: map_id={map_id}, "
                            f"size={len(result['binary'])} bytes"
                        )
                    
                    # Update status to success
                    await self.adapter.update_job_status(map_id, 'success', None)
                    logger.info(f"Processing succeeded: map_id={map_id}")
                    
                except Exception as e:
                    # Update status to failed with error
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    await self.adapter.update_job_status(map_id, 'failed', error_msg)
                    logger.error(
                        f"Processing failed: map_id={map_id}, error={error_msg}",
                        exc_info=True
                    )
                
                finally:
                    # Mark job as done
                    self.queue.task_done()
                    logger.debug(f"Job completed: map_id={map_id}")
                    
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled - shutting down gracefully")
                break
            except Exception as e:
                # Unexpected error in worker loop
                logger.error(f"Worker loop error: {e}", exc_info=True)
                # Continue processing other jobs (don't crash the worker)
    
    def get_queue_length(self) -> int:
        """
        Get number of pending jobs in queue.
        
        Returns:
            Number of jobs waiting to be processed
        """
        return self.queue.qsize()
    
    async def shutdown(self):
        """
        Gracefully shutdown the worker.
        
        Behavior:
        1. Set shutdown flag to reject new jobs
        2. Cancel worker task
        3. Wait for current job to finish
        4. Clean up resources
        
        This method is idempotent (safe to call multiple times).
        """
        if self._shutdown_flag:
            logger.warning("Shutdown already in progress")
            return
        
        self._shutdown_flag = True
        logger.info("Initiating graceful shutdown")
        
        if self.worker_task is None:
            logger.info("No worker to shutdown")
            return
        
        # Cancel worker task
        self.worker_task.cancel()
        
        try:
            # Wait for worker to finish current job
            await self.worker_task
        except asyncio.CancelledError:
            logger.info("Worker task cancelled successfully")
        
        self.worker_task = None
        logger.info("SLAM job queue shutdown complete")
