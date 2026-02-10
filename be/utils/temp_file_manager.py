"""Temporary file manager for rtabmap-reprocess database files.

This module provides a context manager for safely handling temporary .db files
needed by the rtabmap-reprocess CLI tool. It ensures automatic cleanup even if
exceptions occur during processing.
"""

import tempfile
import logging
import glob
import os
import time
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TempFileManager:
    """Context manager for handling rtabmap-reprocess temporary files.
    
    Creates a temporary directory with a unique prefix based on map_id.
    Provides methods to write/read .db files and get paths for CLI invocation.
    Automatically cleans up the temporary directory on context exit.
    
    Example:
        with TempFileManager(map_id='map_123') as temp_mgr:
            temp_mgr.write_db('input.db', input_bytes)
            input_path = temp_mgr.get_path('input.db')
            # Run rtabmap-reprocess with input_path
            output_bytes = temp_mgr.read_db('output.db')
        # Temp directory automatically deleted here
    """
    
    def __init__(self, map_id: str):
        """Initialize TempFileManager.
        
        Args:
            map_id: Unique identifier for the map (used in temp directory prefix)
        """
        self.map_id = map_id
        self.temp_dir: Optional[Path] = None
        self.temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    
    def __enter__(self) -> "TempFileManager":
        """Enter context manager: create temporary directory.
        
        Returns:
            Self for use in 'with' statement
        """
        self.temp_dir_obj = tempfile.TemporaryDirectory(
            prefix=f"rtabmap_{self.map_id}_",
            dir="/tmp"
        )
        self.temp_dir = Path(self.temp_dir_obj.__enter__())
        logger.info(f"Created temp directory: {self.temp_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager: cleanup temporary directory.
        
        Ensures cleanup happens even if an exception occurred during processing.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
            
        Returns:
            False to not suppress exceptions
        """
        if self.temp_dir_obj:
            self.temp_dir_obj.__exit__(exc_type, exc_val, exc_tb)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        return False  # Don't suppress exceptions
    
    def write_db(self, filename: str, data: bytes) -> None:
        """Write binary data to a .db file in the temp directory.
        
        Args:
            filename: Name of the file (e.g., 'input.db', 'output.db')
            data: Binary data to write
            
        Raises:
            RuntimeError: If context manager not entered
            IOError: If write fails
        """
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager context not entered")
        
        file_path = self.temp_dir / filename
        file_path.write_bytes(data)
        logger.info(f"Wrote {filename} to temp directory ({len(data)} bytes)")
    
    def read_db(self, filename: str) -> bytes:
        """Read binary data from a .db file in the temp directory.
        
        Args:
            filename: Name of the file to read (e.g., 'output.db')
            
        Returns:
            Binary data from the file
            
        Raises:
            RuntimeError: If context manager not entered
            FileNotFoundError: If file doesn't exist
            IOError: If read fails
        """
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager context not entered")
        
        file_path = self.temp_dir / filename
        data = file_path.read_bytes()
        logger.info(f"Read {filename} from temp directory ({len(data)} bytes)")
        return data
    
    def get_path(self, filename: str) -> str:
        """Get the absolute path for a file in the temp directory.
        
        Used to pass file paths to external CLI tools like rtabmap-reprocess.
        
        Args:
            filename: Name of the file (e.g., 'input.db', 'output.db')
            
        Returns:
            Absolute path as string
            
        Raises:
            RuntimeError: If context manager not entered
        """
        if self.temp_dir is None:
            raise RuntimeError("TempFileManager context not entered")
        
        return str(self.temp_dir / filename)


def cleanup_orphaned_temps(max_age_hours: float = 1.0) -> int:
    """Delete /tmp/rtabmap_* directories older than max_age_hours.
    
    Useful for cleaning up temporary directories that may have been left behind
    if the application crashed or was forcefully terminated.
    
    Args:
        max_age_hours: Maximum age in hours before deletion (default: 1.0)
        
    Returns:
        Number of directories deleted
    """
    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0
    
    for temp_dir in glob.glob("/tmp/rtabmap_*"):
        try:
            if os.path.getmtime(temp_dir) < cutoff_time:
                shutil.rmtree(temp_dir)
                logger.info(f"Deleted orphaned temp dir: {temp_dir}")
                deleted_count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {temp_dir}: {e}")
    
    logger.info(f"Cleanup complete. Deleted {deleted_count} orphaned directories.")
    return deleted_count
