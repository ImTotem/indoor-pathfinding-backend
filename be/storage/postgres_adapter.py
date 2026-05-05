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
import os
import uuid
from pathlib import Path
from typing import List, Optional

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
        self.storage_root = Path(
            os.getenv(
                "STORAGE_ROOT",
                Path(__file__).resolve().parents[2] / "var" / "storage",
            )
        )
    
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
        async def _fetch_v2():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT storage_path
                    FROM scan_ingest
                    WHERE scan_id = $1
                    """,
                    uuid.UUID(session_id),
                )
                if not row:
                    return None
                return str(self.storage_root / row["storage_path"] / "rtabmap.db")

        async def _fetch_legacy():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT file_path FROM scan_sessions WHERE id = $1",
                    uuid.UUID(session_id)
                )
                return row["file_path"] if row else None
        
        try:
            return await self._retry(_fetch_v2)
        except asyncpg.UndefinedTableError:
            return await self._retry(_fetch_legacy)
    
    async def get_sessions_by_building_id(self, building_id: str) -> List[dict]:
        """
        Fetch all scan_sessions belonging to a building.
        
        Args:
            building_id: buildings.id (UUID string)
        
        Returns:
            List of session dicts, or empty list if none found
        """
        async def _fetch_v2():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT fs.scan_id AS id,
                           bf.building_id,
                           fs.file_name,
                           fs.file_size,
                           fs.status AS floor_scan_status,
                           fs.created_at,
                           si.storage_path,
                           si.build_state,
                           si.ingested_at,
                           ss.keyframe_count
                    FROM building_floor bf
                    JOIN floor_scan fs ON fs.floor_id = bf.floor_id
                    JOIN scan_ingest si ON si.scan_id = fs.scan_id
                    LEFT JOIN scan_session ss ON ss.scan_id = si.scan_id
                    WHERE bf.building_id = $1
                      AND fs.active = true
                    ORDER BY bf.level ASC, fs.created_at ASC
                    """,
                    uuid.UUID(building_id),
                )
                results = []
                for row in rows:
                    status_value = _map_v2_build_state(row["build_state"])
                    storage_path = row["storage_path"]
                    results.append(
                        {
                            "id": str(row["id"]),
                            "building_id": str(row["building_id"]),
                            "file_name": row["file_name"] or "rtabmap.db",
                            "file_path": str(self.storage_root / storage_path / "rtabmap.db"),
                            "file_size": row["file_size"],
                            "status": status_value,
                            "error_message": None,
                            "total_nodes": row["keyframe_count"] or 0,
                            "total_distance": 0.0,
                            "created_at": row["created_at"] or row["ingested_at"],
                            "updated_at": row["ingested_at"],
                        }
                    )
                return results

        async def _fetch_legacy():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, building_id, file_name, file_path, file_size,
                           status, error_message, total_nodes, total_distance,
                           created_at, updated_at
                    FROM scan_sessions
                    WHERE building_id = $1
                    ORDER BY created_at ASC
                    """,
                    uuid.UUID(building_id)
                )
                results = []
                for row in rows:
                    result = dict(row)
                    result["id"] = str(result["id"])
                    result["building_id"] = str(result["building_id"])
                    results.append(result)
                return results
        
        try:
            return await self._retry(_fetch_v2)
        except asyncpg.UndefinedTableError:
            return await self._retry(_fetch_legacy)
    
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

    async def ensure_path_nodes_schema(self):
        """Ensure path_nodes/POI schema exists for shared Spring schema compatibility."""
        async def _create():
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS path_nodes (
                        id UUID PRIMARY KEY,
                        floor_id UUID NOT NULL,
                        x DOUBLE PRECISION NOT NULL,
                        y DOUBLE PRECISION NOT NULL,
                        z DOUBLE PRECISION NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        poi_name VARCHAR(255),
                        poi_category VARCHAR(100),
                        vertical_passage_id UUID,
                        is_passage_entry BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_path_nodes_floor ON path_nodes (floor_id);
                    CREATE INDEX IF NOT EXISTS idx_path_nodes_poi_name ON path_nodes (poi_name);
                    CREATE INDEX IF NOT EXISTS idx_path_nodes_coordinates ON path_nodes (x, y, z);
                    """
                )
        await self._retry(_create)

    async def get_nearest_pois(
        self,
        floor_id: str,
        x: float,
        y: float,
        z: float,
        max_distance: float = 5.0,
        limit: int = 10
    ) -> List[dict]:
        """Get nearby POI nodes for a floor + position."""
        async def _fetch():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, floor_id, x, y, z, poi_name, poi_category,
                           SQRT(POWER(x - $1, 2) + POWER(y - $2, 2) + POWER(z - $3, 2)) AS distance
                    FROM path_nodes
                    WHERE floor_id = $4
                      AND poi_name IS NOT NULL
                      AND SQRT(POWER(x - $1, 2) + POWER(y - $2, 2) + POWER(z - $3, 2)) <= $5
                    ORDER BY distance ASC
                    LIMIT $6
                    """,
                    x, y, z, uuid.UUID(floor_id), max_distance, limit
                )
                return [
                    {
                        "id": str(r["id"]),
                        "floor_id": str(r["floor_id"]),
                        "x": float(r["x"]),
                        "y": float(r["y"]),
                        "z": float(r["z"]),
                        "poi_name": r["poi_name"],
                        "poi_category": r["poi_category"],
                        "distance": float(r["distance"]),
                    }
                    for r in rows
                ]
        return await self._retry(_fetch)

    async def get_floor_maps(self, building_id: str) -> List[dict]:
        """
        Fetch all floor merged DB paths for a building.

        Joins floors + merged_scans to find localization-ready DBs.

        Returns:
            List of dicts with floor_id, floor_name, level, file_path
            sorted by level ASC. Empty list if none found.
        """
        async def _fetch_v2():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT bf.floor_id, bf.name AS floor_name, bf.level,
                           fs.scan_id, si.storage_path
                    FROM building_floor bf
                    JOIN floor_scan fs ON fs.floor_id = bf.floor_id
                    JOIN scan_ingest si ON si.scan_id = fs.scan_id
                    WHERE bf.building_id = $1
                      AND fs.active = true
                    ORDER BY bf.level ASC, fs.created_at ASC
                    """,
                    uuid.UUID(building_id),
                )
                return [
                    {
                        "floor_id": str(r["floor_id"]),
                        "floor_name": r["floor_name"],
                        "level": r["level"],
                        "file_path": str(self.storage_root / r["storage_path"] / "rtabmap.db"),
                        "scan_id": str(r["scan_id"]),
                    }
                    for r in rows
                ]

        async def _fetch_legacy():
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT f.id AS floor_id, f.name AS floor_name, f.level,
                           ms.file_path
                    FROM floors f
                    JOIN merged_scans ms ON ms.floor_id = f.id
                    WHERE f.building_id = $1
                      AND ms.file_path IS NOT NULL
                    ORDER BY f.level ASC
                    """,
                    uuid.UUID(building_id)
                )
                return [
                    {
                        "floor_id": str(r["floor_id"]),
                        "floor_name": r["floor_name"],
                        "level": r["level"],
                        "file_path": r["file_path"],
                    }
                    for r in rows
                ]
        try:
            return await self._retry(_fetch_v2)
        except asyncpg.UndefinedTableError:
            return await self._retry(_fetch_legacy)

    async def get_preview_image_path(self, building_id: str) -> Optional[str]:
        """Get latest available preview image path for a building's completed scan session."""
        async def _fetch():
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT preview_image_path
                    FROM scan_sessions
                    WHERE building_id = $1
                      AND preview_image_path IS NOT NULL
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    uuid.UUID(building_id)
                )
                return row["preview_image_path"] if row else None
        return await self._retry(_fetch)


def _map_v2_build_state(value: str | None) -> str:
    if value == "succeeded":
        return SCAN_STATUS_COMPLETED
    if value in {"pending", "running"}:
        return SCAN_STATUS_PROCESSING
    if value in {"failed", "cancelled"}:
        return SCAN_STATUS_FAILED
    return SCAN_STATUS_UPLOADED
