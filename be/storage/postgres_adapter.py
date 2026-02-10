"""
PostgreSQL adapter for SLAM service.
Uses existing scan_sessions table from indoor-pathfinding-backend (Spring Boot JPA).

Table: scan_sessions
Columns: id (UUID PK), building_id (UUID FK), file_name, file_path, file_size,
         status (ENUM: UPLOADED/EXTRACTING/PROCESSING/COMPLETED/FAILED),
         error_message, preview_image_path, processed_preview_path,
         total_nodes, total_distance, created_at, updated_at

IMPORTANT: No schema modifications allowed. Read/write to existing tables only.
"""

import asyncio
import logging
import uuid
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

# Status values matching ScanStatus enum in Spring Boot
SCAN_STATUS_UPLOADED = "UPLOADED"
SCAN_STATUS_EXTRACTING = "EXTRACTING"
SCAN_STATUS_PROCESSING = "PROCESSING"
SCAN_STATUS_COMPLETED = "COMPLETED"
SCAN_STATUS_FAILED = "FAILED"


class PostgresAdapter:
    """
    Async database adapter using existing scan_sessions table.
    
    Uses asyncpg connection pool (injected via constructor).
    All queries target the scan_sessions table owned by Spring Boot backend.
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def _retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry on connection errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (asyncpg.exceptions.PostgresError, ConnectionRefusedError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached. Last error: {e}")
                    raise
                delay = 0.1 * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def get_session(self, session_id: str) -> dict:
        """
        Fetch scan_session by id.
        
        Args:
            session_id: scan_sessions.id (UUID string)
        
        Returns:
            dict with session info, or empty dict if not found
        """
        async def _fetch():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, building_id, file_name, file_path, file_size,
                           status, error_message, total_nodes, total_distance,
                           created_at, updated_at
                    FROM scan_sessions
                    WHERE id = $1
                    """,
                    uuid.UUID(session_id)
                )
                if row:
                    result = dict(row)
                    result["id"] = str(result["id"])
                    result["building_id"] = str(result["building_id"])
                    return result
                return {}
        
        return await self._retry(_fetch)
    
    async def update_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """
        Update scan_sessions.status and optionally error_message.
        
        Args:
            session_id: scan_sessions.id (UUID string)
            status: One of UPLOADED, EXTRACTING, PROCESSING, COMPLETED, FAILED
            error_message: Error details (for FAILED status)
        """
        async def _update():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE scan_sessions
                    SET status = $1, error_message = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $3
                    """,
                    status,
                    error_message,
                    uuid.UUID(session_id)
                )
        
        await self._retry(_update)
        logger.info(f"Updated session status: session_id={session_id}, status={status}")
    
    async def update_processing_result(
        self,
        session_id: str,
        total_nodes: int,
        total_distance: float,
        preview_image_path: Optional[str] = None,
        processed_preview_path: Optional[str] = None,
    ):
        """
        Update scan_sessions with SLAM processing results and set status to COMPLETED.
        Mirrors ScanSession.updateProcessingResult() in Spring Boot.
        
        Args:
            session_id: scan_sessions.id (UUID string)
            total_nodes: Number of nodes extracted
            total_distance: Total path distance
            preview_image_path: Path to preview image (optional)
            processed_preview_path: Path to processed preview (optional)
        """
        async def _update():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE scan_sessions
                    SET status = $1,
                        total_nodes = $2,
                        total_distance = $3,
                        preview_image_path = $4,
                        processed_preview_path = $5,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $6
                    """,
                    SCAN_STATUS_COMPLETED,
                    total_nodes,
                    total_distance,
                    preview_image_path,
                    processed_preview_path,
                    uuid.UUID(session_id)
                )
        
        await self._retry(_update)
        logger.info(f"Recorded processing result: session_id={session_id}, nodes={total_nodes}")
    
    async def get_file_path(self, session_id: str) -> Optional[str]:
        """
        Get the uploaded .db file_path for a session.
        
        Args:
            session_id: scan_sessions.id (UUID string)
        
        Returns:
            file_path string, or None if not found
        """
        async def _fetch():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT file_path FROM scan_sessions WHERE id = $1",
                    uuid.UUID(session_id)
                )
                return row["file_path"] if row else None
        
        return await self._retry(_fetch)
    
    async def health_check(self) -> str:
        """
        Check PostgreSQL connectivity.
        
        Returns:
            "connected" or error string
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return "connected"
        except Exception as e:
            return f"error: {str(e)}"
