"""
PostgreSQL adapter for SLAM database storage.
Provides async CRUD operations for slam_jobs and slam_databases tables.
"""

import asyncio
import logging
import uuid
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


class PostgresAdapter:
    """
    Async database adapter for SLAM storage operations.
    
    Uses asyncpg connection pool (injected via constructor).
    Implements retry logic for connection failures with exponential backoff.
    """
    
    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize adapter with asyncpg connection pool.
        
        Args:
            pool: asyncpg.Pool instance (managed by FastAPI lifespan)
        """
        self.pool = pool
    
    async def _retry_on_connection_error(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Retry function on connection errors with exponential backoff.
        
        Args:
            func: Async function to retry
            max_retries: Maximum retry attempts (default: 3)
            *args, **kwargs: Arguments passed to func
        
        Returns:
            Result of func
        
        Raises:
            PostgresError or ConnectionRefusedError after max retries
        """
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (asyncpg.exceptions.PostgresError, ConnectionRefusedError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached. Last error: {e}")
                    raise
                delay = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def create_job(self, session_id: str) -> str:
        """
        Create a new SLAM job with 'pending' status.
        
        Args:
            session_id: Session identifier
        
        Returns:
            map_id: Generated UUID (hex format, no dashes)
        
        Raises:
            PostgresError: Database operation failed
        """
        map_id = uuid.uuid4().hex  # Generate UUID without dashes
        
        async def _insert_job():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO slam_jobs (session_id, map_id, status)
                    VALUES ($1, $2, $3)
                    """,
                    session_id,
                    map_id,
                    'pending'
                )
        
        await self._retry_on_connection_error(_insert_job)
        logger.info(f"Created job: session_id={session_id}, map_id={map_id}")
        return map_id
    
    async def update_job_status(
        self, 
        map_id: str, 
        status: str, 
        error_message: Optional[str] = None
    ):
        """
        Update job status and error message.
        
        Args:
            map_id: Map identifier
            status: New status ('pending', 'in_progress', 'success', 'failed')
            error_message: Error details (only for 'failed' status)
        
        Raises:
            PostgresError: Database operation failed
        """
        async def _update_status():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE slam_jobs
                    SET status = $1, error_message = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE map_id = $3
                    """,
                    status,
                    error_message,
                    map_id
                )
        
        await self._retry_on_connection_error(_update_status)
        logger.info(f"Updated job status: map_id={map_id}, status={status}")
    
    async def get_job_status(self, map_id: str) -> dict:
        """
        Fetch job information by map_id.
        
        Args:
            map_id: Map identifier
        
        Returns:
            dict with keys: map_id, session_id, status, created_at, updated_at, error_message
            Returns empty dict if job not found
        
        Raises:
            PostgresError: Database operation failed
        """
        async def _fetch_job():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT map_id, session_id, status, created_at, updated_at, error_message
                    FROM slam_jobs
                    WHERE map_id = $1
                    """,
                    map_id
                )
                if row:
                    return dict(row)
                return {}
        
        result = await self._retry_on_connection_error(_fetch_job)
        logger.debug(f"Fetched job status: map_id={map_id}")
        return result
    
    async def store_input_db(self, map_id: str, db_bytes: bytes):
        """
        Store input.db binary data.
        
        Args:
            map_id: Map identifier
            db_bytes: Binary database file content
        
        Raises:
            PostgresError: Database operation failed
        """
        size_bytes = len(db_bytes)
        
        async def _insert_db():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO slam_databases (map_id, db_type, db_data, size_bytes)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (map_id, db_type) 
                    DO UPDATE SET db_data = EXCLUDED.db_data, size_bytes = EXCLUDED.size_bytes
                    """,
                    map_id,
                    'input',
                    db_bytes,
                    size_bytes
                )
        
        await self._retry_on_connection_error(_insert_db)
        logger.info(f"Stored input.db: map_id={map_id}, size={size_bytes} bytes")
    
    async def store_output_db(self, map_id: str, db_bytes: bytes):
        """
        Store output.db binary data.
        
        Args:
            map_id: Map identifier
            db_bytes: Binary database file content
        
        Raises:
            PostgresError: Database operation failed
        """
        size_bytes = len(db_bytes)
        
        async def _insert_db():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO slam_databases (map_id, db_type, db_data, size_bytes)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (map_id, db_type) 
                    DO UPDATE SET db_data = EXCLUDED.db_data, size_bytes = EXCLUDED.size_bytes
                    """,
                    map_id,
                    'output',
                    db_bytes,
                    size_bytes
                )
        
        await self._retry_on_connection_error(_insert_db)
        logger.info(f"Stored output.db: map_id={map_id}, size={size_bytes} bytes")
    
    async def fetch_input_db(self, map_id: str) -> bytes:
        """
        Retrieve input.db binary data.
        
        Args:
            map_id: Map identifier
        
        Returns:
            Binary database file content (empty bytes if not found)
        
        Raises:
            PostgresError: Database operation failed
        """
        async def _fetch_db():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT db_data FROM slam_databases
                    WHERE map_id = $1 AND db_type = $2
                    """,
                    map_id,
                    'input'
                )
                if row:
                    return bytes(row['db_data'])
                return b''
        
        result = await self._retry_on_connection_error(_fetch_db)
        logger.debug(f"Fetched input.db: map_id={map_id}, size={len(result)} bytes")
        return result
    
    async def fetch_output_db(self, map_id: str) -> bytes:
        """
        Retrieve output.db binary data.
        
        Args:
            map_id: Map identifier
        
        Returns:
            Binary database file content (empty bytes if not found)
        
        Raises:
            PostgresError: Database operation failed
        """
        async def _fetch_db():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT db_data FROM slam_databases
                    WHERE map_id = $1 AND db_type = $2
                    """,
                    map_id,
                    'output'
                )
                if row:
                    return bytes(row['db_data'])
                return b''
        
        result = await self._retry_on_connection_error(_fetch_db)
        logger.debug(f"Fetched output.db: map_id={map_id}, size={len(result)} bytes")
        return result
