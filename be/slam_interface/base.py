# slam_interface/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Callable, Optional, List

class SLAMEngineBase(ABC):
    """SLAM 엔진 추상 인터페이스"""
    
    @abstractmethod
    async def process(
        self,
        session_id: str,
        frames_data: Dict,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """맵 생성"""
        pass
    
    @abstractmethod
    async def localize(
        self,
        map_id: str,
        images: List[bytes],
        intrinsics: Optional[Dict] = None,
        initial_pose: Optional[Dict] = None
    ) -> Dict:
        """위치 추정"""
        pass
    
    @abstractmethod
    def extract_intrinsics_from_db(self, db_path: str) -> dict:
        """Extract camera intrinsics from map database"""
        pass
    
    @abstractmethod
    def scale_intrinsics(self, original: dict, new_width: int, new_height: int) -> dict:
        """Scale camera intrinsics for resolution change"""
        pass
    
    @abstractmethod
    def save_map(self, map_data: Dict, map_id: str, base_dir: Path) -> Path:
        """
        Save map in engine-specific format.
        
        Args:
            map_data: Map data dict with 'binary' and 'metadata' keys
            map_id: Unique identifier for the map
            base_dir: Base directory for map storage (e.g., /data/maps)
        
        Returns:
            Path to saved map file (engine determines extension)
        """
        pass
    
    @abstractmethod
    def load_map(self, map_id: str, base_dir: Path) -> bytes:
        """
        Load map from engine-specific format.
        
        Args:
            map_id: Unique identifier for the map
            base_dir: Base directory for map storage
        
        Returns:
            Binary map data in engine-specific format
        """
        pass

