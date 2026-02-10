# storage/storage_manager.py
import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
import aiofiles
from datetime import datetime

from config.settings import settings
from slam_interface.base import SLAMEngineBase

class StorageManager:
    """세션 및 맵 데이터 저장소 관리자"""
    
    def __init__(self):
        self.base_dir = settings.DATA_DIR
        self.sessions_dir = settings.SESSIONS_DIR
        self.maps_dir = settings.MAPS_DIR
        
        # 디렉토리 생성
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_session(self, session_id: str, device_info: Dict) -> Dict:
        """새 스캔 세션 생성"""
        session_path = self.sessions_dir / session_id
        session_path.mkdir(exist_ok=True)
        
        (session_path / "images").mkdir(exist_ok=True)
        (session_path / "chunks").mkdir(exist_ok=True)
        
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "device_info": device_info,
            "status": "scanning",
            "total_frames": 0,
            "total_chunks": 0,
        }
        
        async with aiofiles.open(session_path / "metadata.json", "w") as f:
            await f.write(json.dumps(metadata, indent=2))
        
        return metadata
    
    async def save_chunk(
        self, 
        session_id: str, 
        chunk_index: int, 
        frames: List[Dict]
    ) -> int:
        """청크 단위로 프레임 저장"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        images_dir = session_path / "images"
        poses = []
        
        for i, frame in enumerate(frames):
            frame_idx = chunk_index * 1000 + i
            
            try:
                image_data = base64.b64decode(frame["image"])
                
                image_path = images_dir / f"{frame_idx:06d}.jpg"
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(image_data)
                
                poses.append({
                    "frame_index": frame_idx,
                    "timestamp": frame["timestamp"],
                    "position": frame["position"],
                    "orientation": frame["orientation"],
                    "image_path": str(image_path.relative_to(session_path)),
                    "imu": frame.get("imu"),
                    "camera_intrinsics": frame.get("camera_intrinsics"),
                })
                
            except Exception as e:
                print(f"프레임 {frame_idx} 저장 실패: {e}")
                continue
        
        chunk_file = session_path / "chunks" / f"chunk_{chunk_index:04d}.json"
        async with aiofiles.open(chunk_file, "w") as f:
            await f.write(json.dumps(poses, indent=2))
        
        await self._update_metadata(session_id, len(poses), chunk_index)
        
        return len(poses)
    
    async def save_frame_binary(
        self,
        session_id: str,
        chunk_index: int,
        frame_index: int,
        frame_data: Dict
    ) -> None:
        """바이너리 프레임 저장 (base64 디코딩 없음)"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        images_dir = session_path / "images"
        depth_dir = session_path / "depth"
        depth_dir.mkdir(exist_ok=True)
        
        frame_idx = chunk_index * 1000 + frame_index
        
        image_data = frame_data.get("image_data")
        if not image_data:
            raise ValueError("image_data is required")
        
        image_path = images_dir / f"{frame_idx:06d}.jpg"
        async with aiofiles.open(image_path, "wb") as f:
            await f.write(image_data)
        
        depth_path = None
        depth_data = frame_data.get("depth_data")
        depth_width = frame_data.get("depth_width")
        depth_height = frame_data.get("depth_height")
        
        if depth_data and depth_width and depth_height:
            import numpy as np
            import cv2
            
            depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_height, depth_width))
            depth_path = depth_dir / f"{frame_idx:06d}.png"
            
            cv2.imwrite(str(depth_path), depth_array)
        
        pose_entry = {
            "frame_index": frame_idx,
            "timestamp": frame_data.get("timestamp", 0),
            "position": frame_data.get("position", [0, 0, 0]),
            "orientation": frame_data.get("orientation", [0, 0, 0, 1]),
            "image_path": str(image_path.relative_to(session_path)),
            "depth_path": str(depth_path.relative_to(session_path)) if depth_path else None,
            "imu": frame_data.get("imu_data"),
            "camera_intrinsics": frame_data.get("camera_intrinsics"),
        }
        
        chunk_file = session_path / "chunks" / f"chunk_{chunk_index:04d}.json"
        
        if chunk_file.exists():
            async with aiofiles.open(chunk_file, "r") as f:
                poses = json.loads(await f.read())
        else:
            poses = []
        
        poses.append(pose_entry)
        
        async with aiofiles.open(chunk_file, "w") as f:
            await f.write(json.dumps(poses, indent=2))
        
        await self._update_metadata(session_id, 1, chunk_index)
    
    async def _update_metadata(
        self, 
        session_id: str, 
        new_frames: int,
        chunk_index: int
    ):
        """메타데이터 업데이트"""
        metadata_path = self.sessions_dir / session_id / "metadata.json"
        
        async with aiofiles.open(metadata_path, "r") as f:
            metadata = json.loads(await f.read())
        
        metadata["total_frames"] += new_frames
        metadata["total_chunks"] = max(metadata["total_chunks"], chunk_index + 1)
        metadata["last_updated"] = datetime.now().isoformat()
        
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(json.dumps(metadata, indent=2))
    
    async def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부"""
        return (self.sessions_dir / session_id).exists()
    
    async def get_session_status(self, session_id: str) -> Dict:
        """세션 상태 조회"""
        metadata_path = self.sessions_dir / session_id / "metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        async with aiofiles.open(metadata_path, "r") as f:
            metadata = json.loads(await f.read())
        
        return metadata
    
    async def update_status(
        self, 
        session_id: str, 
        status: str, 
        **kwargs
    ):
        """세션 상태 업데이트"""
        metadata_path = self.sessions_dir / session_id / "metadata.json"
        
        async with aiofiles.open(metadata_path, "r") as f:
            metadata = json.loads(await f.read())
        
        metadata["status"] = status
        metadata["updated_at"] = datetime.now().isoformat()
        metadata.update(kwargs)
        
        async with aiofiles.open(metadata_path, "w") as f:
            await f.write(json.dumps(metadata, indent=2))
    
    async def update_progress(self, session_id: str, progress: float):
        """처리 진행률 업데이트"""
        await self.update_status(session_id, "processing", progress=progress)
    
    async def load_session_data(self, session_id: str) -> Dict:
        """세션 데이터 전체 로드"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        chunks_dir = session_path / "chunks"
        all_poses = []
        
        for chunk_file in sorted(chunks_dir.glob("chunk_*.json")):
            async with aiofiles.open(chunk_file, "r") as f:
                chunk_data = json.loads(await f.read())
                all_poses.extend(chunk_data)
        
        return {
            "session_id": session_id,
            "session_path": str(session_path),
            "images_dir": str(session_path / "images"),
            "poses": all_poses,
        }
    
    async def save_map(self, session_id: str, map_data: Dict, engine: SLAMEngineBase) -> str:
        """생성된 맵 저장"""
        map_id = f"map_{session_id}"
        
        if "binary" in map_data:
            engine.save_map(map_data, map_id, self.maps_dir)
        
        meta_path = self.maps_dir / f"{map_id}_meta.json"
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(map_data["metadata"], indent=2))
        
        return map_id
    
    async def load_map(self, map_id: str, engine: SLAMEngineBase) -> Dict:
        """맵 데이터 로드"""
        meta_path = self.maps_dir / f"{map_id}_meta.json"
        
        if not meta_path.exists():
            raise ValueError(f"Map {map_id} not found")
        
        async with aiofiles.open(meta_path, "r") as f:
            metadata = json.loads(await f.read())
        
        binary_data = engine.load_map(map_id, self.maps_dir)
        
        return {
            "map_id": map_id,
            "metadata": metadata,
            "binary": binary_data,
        }

